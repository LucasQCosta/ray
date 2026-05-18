from ray._common.deprecation import deprecation_warning
from ray.rllib.algorithms.ppo_lag.torch.default_ppo_lag_torch_rl_module import (  # noqa
    DefaultPPOTorchRLModule as PPOTorchRLModule,
)

deprecation_warning(
    old="ray.rllib.algorithms.ppo_lag.torch.ppo_torch_rl_module.PPOTorchRLModule",
    new="ray.rllib.algorithms.ppo_lag.torch.default_ppo_torch_rl_module."
    "DefaultPPOTorchRLModule",
    error=False,
)
