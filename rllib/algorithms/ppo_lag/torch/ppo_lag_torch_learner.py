import logging
from typing import Any, Dict

import numpy as np

from ray.rllib.algorithms.ppo_lag.ppo_lag import (
    LEARNER_RESULTS_CURR_KL_COEFF_KEY,
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    PPOConfig,
)
from ray.rllib.algorithms.ppo_lag.ppo_lag_learner import PPOLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import ENTROPY_KEY, POLICY_LOSS_KEY, VF_LOSS_KEY
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.typing import ModuleID, TensorType

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class PPOTorchLearner(PPOLearner, TorchLearner):
    """Implements torch-specific PPO loss logic on top of PPOLearner.

    This class implements the ppo loss under `self.compute_loss_for_module()`.

    PPO-Lag / Lagrangian PPO extensions implemented here:
    - Maintain and update a dual variable (Lagrange multiplier) $\lambda \ge 0$.
    - Learn an additional cost value function $V_c(s)$ (if present in the RLModule).
    - Shape the policy advantage using cost advantages (primal objective).
    - Update $\lambda$ using a separate optimizer (dual ascent/descent step).
    """
    @override(TorchLearner)
    def configure_optimizers_for_module(
        self, module_id: ModuleID, config: PPOConfig
    ) -> None:
        # PPO-Lag uses a primal-dual update.
        #
        # Primal parameters (policy + value nets) are optimized with PPO's loss.
        # The dual variable $\lambda$ is optimized separately from a dual objective
        # derived from the constraint violation (see `lambda_loss` below).
        #
        # We therefore keep `lambda_param` OUT of the default optimizer and give it
        # its own optimizer with its own learning rate (`lambda_lr`).
        module = self._module[module_id]
        params = [
            p
            for (name, p) in module.named_parameters()
            if name != "lambda_param"
        ]
        optimizer = torch.optim.Adam(params)
        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=params,
            lr_or_lr_schedule=config.lr,
        )

        lag_module = self.module[module_id].unwrapped()
        lambda_lr = getattr(config, "lambda_lr", 0.05)
        # Dual optimizer for the Lagrange multiplier parameter $\lambda$.
        lambda_optimizer = torch.optim.Adam([lag_module.lambda_param], lr=lambda_lr)
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="lambda_optimizer",
            optimizer=lambda_optimizer,
            params=[lag_module.lambda_param],
            lr_or_lr_schedule=lambda_lr,
        )


    @override(TorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        module = self.module[module_id].unwrapped()

        # Possibly apply masking to some sub loss terms and to the total loss term
        # at the end. Masking could be used for RNN-based model (zero padded `batch`)
        # and for PPO's batched value function (and bootstrap value) computations,
        # for which we add an (artificial) timestep to each episode to
        # simplify the actual computation.
        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            mask = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
            mask = mask.bool()
            num_valid = torch.sum(mask)

            def possibly_masked_mean(data_):
                data_ = data_ if torch.is_tensor(data_) else torch.as_tensor(data_)
                mask_ = mask.to(device=data_.device)
                num_valid_ = num_valid.to(device=data_.device)
                if num_valid_.item() == 0:
                    return torch.mean(data_)
                return torch.sum(data_[mask_]) / num_valid_

        else:
            def possibly_masked_mean(data_):
                data_ = data_ if torch.is_tensor(data_) else torch.as_tensor(data_)
                return torch.mean(data_)

        # Current dual variable value $\lambda$ (constrained to be non-negative
        # by the RLModule's parameterization, e.g. softplus).
        current_lambda = module.get_lambda()
        
        if "cost_advantages" in batch:
            adv_r = batch[Postprocessing.ADVANTAGES]
            adv_c = batch["cost_advantages"]
            
            # PPO-Lag shapes the reward advantage with a cost advantage term.
            #
            # One practical variant used here is:
            # $$\tilde{A} = \frac{A_r - \lambda A_c}{1 + \lambda}$$
            #
            # Important: we detach $\lambda$ so policy gradients do not flow through
            # the dual update (the dual step is handled by `lambda_optimizer`).
            lambda_detached = current_lambda.detach()
            
            batch[Postprocessing.ADVANTAGES] = (adv_r - lambda_detached * adv_c) / (1.0 + lambda_detached)

        action_dist_class_train = module.get_train_action_dist_cls()
        action_dist_class_exploration = module.get_exploration_action_dist_cls()

        curr_action_dist = action_dist_class_train.from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        # TODO (sven): We should ideally do this in the LearnerConnector (separation of
        #  concerns: Only do things on the EnvRunners that are required for computing
        #  actions, do NOT do anything on the EnvRunners that's only required for a
        #   training update).
        prev_action_dist = action_dist_class_exploration.from_logits(
            batch[Columns.ACTION_DIST_INPUTS]
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if config.use_kl_loss:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = possibly_masked_mean(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = possibly_masked_mean(curr_entropy)

        surrogate_loss = torch.min(
            batch[Postprocessing.ADVANTAGES] * logp_ratio,
            batch[Postprocessing.ADVANTAGES]
            * torch.clamp(logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
        )

        # Compute value function losses.
        # Reward critic:
        # $$L_V = \mathrm{MSE}(V(s_t), \hat{V}_t)$$
        # and RLlib clips the per-sample squared error by `vf_clip_param` before
        # averaging (this is why `vf_loss` often saturates near that clip value).
        #
        # Cost critic (if available):
        # $$L_{V_c} = \mathrm{MSE}(V_c(s_t), \hat{V}^c_t)$$
        if config.use_critic:
            value_fn_out = module.compute_values(
                batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
            )
            vf_loss = torch.pow(value_fn_out - batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
            mean_vf_loss = possibly_masked_mean(vf_loss_clipped)
            mean_vf_unclipped_loss = possibly_masked_mean(vf_loss)
            if "cost_value_targets" in batch:
                cost_fn_out = module.compute_cost_values(
                    batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
                )
                cost_vf_loss = torch.pow(
                    cost_fn_out - batch["cost_value_targets"], 2.0
                )
                mean_cost_vf_loss = possibly_masked_mean(cost_vf_loss)
            else:
                mean_cost_vf_loss = torch.tensor(0.0, device=value_fn_out.device)
        # Ignore the value function -> Set all to 0.0.
        else:
            z = torch.tensor(0.0, device=surrogate_loss.device)
            value_fn_out = mean_vf_unclipped_loss = vf_loss_clipped = mean_vf_loss = z
            mean_cost_vf_loss = z

        total_loss = possibly_masked_mean(
            -surrogate_loss
            + config.vf_loss_coeff * vf_loss_clipped
            - (
                self.entropy_coeff_schedulers_per_module[module_id].get_current_value()
                * curr_entropy
            )
        )

        # Add cost value function loss, if available.
        if mean_cost_vf_loss is not None:
            total_loss = total_loss + config.vf_loss_coeff * mean_cost_vf_loss

        # Always log the cost value loss (0.0 if not applicable).
        self.metrics.log_value(
            (module_id, "cost_vf_loss"),
            float(mean_cost_vf_loss.item())
            if torch.is_tensor(mean_cost_vf_loss)
            else float(mean_cost_vf_loss),
            window=1,
        )

        # Add mean_kl_loss (already processed through `possibly_masked_mean`),
        # if necessary.
        if config.use_kl_loss:
            total_loss += self.curr_kl_coeffs_per_module[module_id] * mean_kl_loss

        if "costs" in batch:
            mean_cost = possibly_masked_mean(batch["costs"])
            cost_limit = getattr(config, "cost_limit", 25.0)
            
            # Dual objective term: encourage $\lambda$ to increase when the
            # expected cost exceeds the limit, and decrease otherwise.
            #
            # Using the batch estimate $\mathbb{E}[C] \approx \text{mean\_cost}$:
            # $$L_\lambda = -\lambda\,(\mathbb{E}[C] - C_{limit})$$
            #
            # Since $\lambda$ is optimized by `lambda_optimizer`, this produces the
            # desired primal-dual behavior in practice.
            lambda_loss = -current_lambda * (mean_cost - cost_limit)
            
            total_loss = total_loss + lambda_loss
            
            self.metrics.log_value((module_id, "mean_cost"), mean_cost.item(), window=1)
            self.metrics.log_value((module_id, "lambda_value"), current_lambda.item(), window=1)
            self.metrics.log_value((module_id, "lambda_loss"), lambda_loss.item(), window=1)

        # Log total loss as a scalar for easier debugging/visualization.
        self.metrics.log_value((module_id, "total_loss"), total_loss.item(), window=1)

        # Log important loss stats.
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -possibly_masked_mean(surrogate_loss),
                VF_LOSS_KEY: mean_vf_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS], value_fn_out
                ),
                ENTROPY_KEY: mean_entropy,
                LEARNER_RESULTS_KL_KEY: mean_kl_loss,
            },
            key=module_id,
            window=1,  # <- single items (should not be mean/ema-reduced over time).
        )
        # Return the total loss.
        return total_loss

    @override(PPOLearner)
    def _update_module_kl_coeff(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        kl_loss: float,
    ) -> None:
        if np.isnan(kl_loss):
            logger.warning(
                f"KL divergence for Module {module_id} is non-finite, this "
                "will likely destabilize your model and the training "
                "process. Action(s) in a specific state have near-zero "
                "probability. This can happen naturally in deterministic "
                "environments where the optimal policy has zero mass for a "
                "specific action. To fix this issue, consider setting "
                "`kl_coeff` to 0.0 or increasing `entropy_coeff` in your "
                "config."
            )

        # Update the KL coefficient.
        curr_var = self.curr_kl_coeffs_per_module[module_id]
        if kl_loss > 2.0 * config.kl_target:
            # TODO (Kourosh) why not 2?
            curr_var.data *= 1.5
        elif kl_loss < 0.5 * config.kl_target:
            curr_var.data *= 0.5

        # Log the updated KL-coeff value.
        self.metrics.log_value(
            (module_id, LEARNER_RESULTS_CURR_KL_COEFF_KEY),
            curr_var.item(),
            window=1,
        )
