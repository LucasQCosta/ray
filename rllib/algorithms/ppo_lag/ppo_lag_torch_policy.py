import logging
from typing import Dict, List, Type, Union

import ray
import numpy as np

from ray.rllib.algorithms.ppo_lag.ppo_lag_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

# PPO-Lag custom batch keys.
COSTS = "costs"
COST_ADVANTAGES = "cost_advantages"
COST_VALUE_TARGETS = "cost_value_targets"

def compute_cost_gae(
    costs: np.ndarray,
    dones: np.ndarray,
    cost_vf_preds: np.ndarray,
    gamma: float,
    lambda_: float,
):
    """Compute GAE for the cost signal.

    This mirrors reward GAE, but uses costs instead of rewards.

    Args:
        costs: per-step costs c_t.
        dones: terminal flags.
        cost_vf_preds: predicted cost values V_c(s_t).
        gamma: cost discount factor.
        lambda_: GAE lambda for cost.

    Returns:
        cost_advantages, cost_value_targets
    """
    costs = costs.astype(np.float32)
    dones = dones.astype(np.float32)
    cost_vf_preds = cost_vf_preds.astype(np.float32)

    cost_advantages = np.zeros_like(costs, dtype=np.float32)
    last_gae_lam = 0.0

    for t in reversed(range(len(costs))):
        if t == len(costs) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = 0.0
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = cost_vf_preds[t + 1]

        delta = costs[t] + gamma * next_value * next_non_terminal - cost_vf_preds[t]
        last_gae_lam = delta + gamma * lambda_ * next_non_terminal * last_gae_lam
        cost_advantages[t] = last_gae_lam

    cost_value_targets = cost_advantages + cost_vf_preds

    return cost_advantages.astype(np.float32), cost_value_targets.astype(np.float32)

class PPOTorchPolicy(
    ValueNetworkMixin,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    KLCoeffMixin,
    TorchPolicyV2,
):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.algorithms.ppo_lag.ppo_lag.PPOConfig().to_dict(), **config)
        validate_config(config)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        # PPO-Lag configuration.
        self.use_cost_threshold = bool(config.get("use_cost_threshold", True))
        self.lambda_cost_scale = float(config.get("lambda_cost_scale", 1.0))
        self.cost_limit = float(config.get("cost_limit", 25.0))
        self.lambda_lr = float(config.get("lambda_lr", 0.05))
        self.lambda_max = float(config.get("lambda_max", 1e6))
        self.cost_gamma = float(config.get("cost_gamma", config.get("gamma", 0.99)))
        self.cost_lambda = float(config.get("cost_lambda", config.get("lambda", 0.95)))
        self.cost_vf_loss_coeff = float(
            config.get("cost_vf_loss_coeff", config.get("vf_loss_coeff", 1.0))
        )
        self.normalize_lagrangian_advantage = bool(
            config.get("normalize_lagrangian_advantage", True)
        )
        self.cost_key = config.get("cost_key", "cost")

        # Lagrange multiplier.
        # We optimize an unconstrained parameter and obtain lambda >= 0 via softplus.
        lambda_init = float(config.get("lambda_init", 0.0))
        lambda_init = np.clip(
            lambda_init,
            0.0,
            self.lambda_max,
        )

        self.lambda_param = torch.nn.Parameter(
            torch.tensor(lambda_init, dtype=torch.float32)
        )

        self.lambda_optimizer = torch.optim.Adam(
            [self.lambda_param],
            lr=self.lambda_lr,
        )

        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # PPO-Lag: get per-step costs and cost advantages.
        # These are added in postprocess_trajectory().
        if COSTS in train_batch:
            costs = train_batch[COSTS].float().to(logits.device)
        else:
            costs = torch.zeros_like(train_batch[Postprocessing.ADVANTAGES]).to(logits.device)

        if COST_ADVANTAGES in train_batch:
            cost_advantages = train_batch[COST_ADVANTAGES].float().to(logits.device)
        else:
            cost_advantages = costs


        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        reward_advantages = train_batch[Postprocessing.ADVANTAGES].float().to(logp_ratio.device) # Ar 

        current_lambda_detached = (
            self.lambda_param
            .detach()
            .to(logp_ratio.device)
            .clamp(0.0, self.lambda_max)
        ) # lambda

        lagrangian_advantages = (
            reward_advantages - current_lambda_detached * cost_advantages
        ) # A_lag = A_r - lambda * A_c

        if self.normalize_lagrangian_advantage:
            lagrangian_advantages = lagrangian_advantages / (
                1.0 + current_lambda_detached
            )

        surrogate_loss = torch.min(
            lagrangian_advantages * logp_ratio,
            lagrangian_advantages
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        ) # PPO-Lag Loss using: L = min(A_lag * r, A_lag * clip(r))

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        # PPO-Lag: optional cost value function loss.
        # For this to be active, your model must implement:
        #     def cost_value_function(self): ...
        if (
            self.config["use_critic"]
            and hasattr(model, "cost_value_function")
            and COST_VALUE_TARGETS in train_batch
        ):
            cost_value_fn_out = model.cost_value_function()
            cost_value_targets = train_batch[COST_VALUE_TARGETS].float().to(
                cost_value_fn_out.device
            )

            cost_vf_loss = torch.pow(cost_value_fn_out - cost_value_targets, 2.0)
            cost_vf_loss_clipped = torch.clamp(
                cost_vf_loss, 0, self.config["vf_clip_param"]
            )
            mean_cost_vf_loss = reduce_mean_valid(cost_vf_loss_clipped)
        else:
            cost_value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            mean_cost_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)        

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add cost critic loss if available.
        total_loss = total_loss + self.cost_vf_loss_coeff * mean_cost_vf_loss

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        value_targets = train_batch[
            Postprocessing.VALUE_TARGETS
        ]

        if (
            self.config["use_critic"]
            and value_targets.numel() > 1
        ):
            vf_explained_var = explained_variance(
                value_targets,
                value_fn_out,
            )
        else:
            vf_explained_var = torch.tensor(
                0.0,
                dtype=value_fn_out.dtype,
                device=value_fn_out.device,
            )

        model.tower_stats["vf_explained_var"] = (
            vf_explained_var
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        # PPO-Lag stats and lambda update.
        minibatch_mean_cost = reduce_mean_valid(costs)

        lambda_for_stats = (
            self.lambda_param
            .detach()
            .to(logp_ratio.device)
            .clamp(0.0, self.lambda_max)
        )

        model.tower_stats["mean_cost"] = minibatch_mean_cost
        model.tower_stats["mean_cost_vf_loss"] = mean_cost_vf_loss
        model.tower_stats["constraint_lambda"] = lambda_for_stats
        model.tower_stats["lagrangian_advantage_mean"] = reduce_mean_valid(
            lagrangian_advantages
        )

        return total_loss
    
    def update_lagrange_multiplier(self, observed_cost: float) -> float:
        """Update lambda once using the full training-batch mean cost."""

        if not np.isfinite(observed_cost):
            raise ValueError(
                f"observed_cost must be finite, received {observed_cost}"
            )
        
        if self.use_cost_threshold:
            # Original PPO-Lag behavior: update from J_C - d.
            lambda_update_signal = observed_cost - self.cost_limit
        else:
            # No-threshold test: update from J_C directly.
            lambda_update_signal = self.lambda_cost_scale * observed_cost

        lambda_update_signal = torch.tensor(
            lambda_update_signal,
            dtype=self.lambda_param.dtype,
            device=self.lambda_param.device,
        )

        lambda_loss = -self.lambda_param * lambda_update_signal

        # cost_difference = torch.tensor(
        #     observed_cost - self.cost_limit, # [Jc - threshold]
        #     dtype=self.lambda_param.dtype,
        #     device=self.lambda_param.device,
        # )

        # # Minimizing this loss increases lambda when cost > limit
        # # and decreases lambda when cost < limit.
        # lambda_loss = -self.lambda_param * cost_difference

        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()

        # Project lambda onto [0, lambda_max].
        with torch.no_grad():
            self.lambda_param.clamp_(
                min=0.0,
                max=self.lambda_max,
            )

        return float(
            self.lambda_param.detach().cpu().item()
        )
        
    @override(TorchPolicyV2)
    def get_state(self):
        state = super().get_state()

        state["ppo_lag_lambda"] = float(
            self.lambda_param.detach().cpu().item()
        )

        state["ppo_lag_lambda_optimizer"] = (
            self.lambda_optimizer.state_dict()
        )

        return state


    @override(TorchPolicyV2)
    def set_state(self, state):
        super().set_state(state)

        if "ppo_lag_lambda" in state:
            with torch.no_grad():
                self.lambda_param.fill_(
                    float(state["ppo_lag_lambda"])
                )

        if "ppo_lag_lambda_optimizer" in state:
            self.lambda_optimizer.load_state_dict(
                state["ppo_lag_lambda_optimizer"]
            )


    # TODO: Make this an event-style subscription (e.g.:
    #  "after_gradients_computed").
    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,

                # PPO-Lag metrics.
                "mean_cost": torch.mean(
                    torch.stack(self.get_tower_stats("mean_cost"))
                ),
                "cost_vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_cost_vf_loss"))
                ),
                "constraint_lambda": torch.mean(
                    torch.stack(self.get_tower_stats("constraint_lambda"))
                ),
                "lagrangian_advantage_mean": torch.mean(
                    torch.stack(self.get_tower_stats("lagrangian_advantage_mean"))
                ),
            }
        )

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # PPO-Lag: extract per-step costs from infos.
        # Your env must return:
        #     infos[agent_id]["cost"] = ...
        infos = sample_batch.get(SampleBatch.INFOS, None)

        if infos is not None:
            costs = np.array(
                [
                    float(info.get(self.cost_key, 0.0)) if isinstance(info, dict) else 0.0
                    for info in infos
                ],
                dtype=np.float32,
            )
        else:
            costs = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)

        sample_batch[COSTS] = costs

        # Reward GAE remains unchanged.
        # Do not modify rewards here; PPO-Lag uses a separate cost advantage.
        with torch.no_grad():
            sample_batch = compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )

        # Cost value predictions.
        #
        # In the old API, obtaining cost value predictions in postprocess is not as
        # straightforward as reward value predictions.
        #
        # For now, this uses zero baseline, which still gives a valid cost-return
        # advantage estimate.
        #
        # If your custom model implements a cost critic, we can later improve this
        # part to use model.cost_value_function() predictions here.
        cost_vf_preds = np.zeros_like(costs, dtype=np.float32)

        dones = sample_batch[SampleBatch.TERMINATEDS]
        cost_advantages, cost_value_targets = compute_cost_gae(
            costs=costs,
            dones=dones,
            cost_vf_preds=cost_vf_preds,
            gamma=self.cost_gamma,
            lambda_=self.cost_lambda,
        )

        sample_batch[COST_ADVANTAGES] = cost_advantages
        sample_batch[COST_VALUE_TARGETS] = cost_value_targets

        return sample_batch