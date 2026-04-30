from ray._common.deprecation import deprecation_warning
from ray.rllib.algorithms.ppo_lag.default_ppo_rl_module import (  # noqa
    DefaultPPORLModule as PPORLModule,
)

deprecation_warning(
    old="ray.rllib.algorithms.ppo_lag.ppo_rl_module.PPORLModule",
    new="ray.rllib.algorithms.ppo_lag.default_ppo_rl_module.DefaultPPORLModule",
    error=False,
)
