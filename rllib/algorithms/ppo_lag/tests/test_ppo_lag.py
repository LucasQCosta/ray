import unittest

import gymnasium as gym

import ray
import ray.rllib.algorithms.ppo_lag as ppo_lag
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.utils.metrics import LEARNER_RESULTS
from ray.rllib.utils.test_utils import check_train_results_new_api_stack
from ray.tune.registry import register_env


class ConstantCostInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, cost: float):
        super().__init__(env)
        self._cost = float(cost)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info is None:
            info = {}
        info["cost"] = self._cost
        return obs, reward, terminated, truncated, info


def _make_cost_env(env_config):
    cost = env_config.get("cost", 0.0)
    env = gym.make("CartPole-v1")
    return ConstantCostInfoWrapper(env, cost=cost)


class TestPPOLagEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init()
        register_env("cartpole_cost_info", _make_cost_env)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def _build_algo(self, *, cost: float, cost_limit: float, lambda_lr: float):
        config = (
            ppo_lag.PPOConfig()
            .framework("torch")
            .environment("cartpole_cost_info", env_config={"cost": cost})
            .env_runners(num_env_runners=0, rollout_fragment_length=32)
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .learners(
                num_learners=0,
                add_default_connectors_to_learner_pipeline=True,
            )
            .training(
                num_epochs=1,
                minibatch_size=32,
                train_batch_size=64,
                use_kl_loss=False,
            )
        )

        # PPO-Lag-specific knobs (currently read via getattr in the learner).
        config.cost_limit = float(cost_limit)
        config.lambda_lr = float(lambda_lr)

        return config.build()

    def test_train_updates_lambda_from_env_cost_info(self):
        algo = self._build_algo(cost=10.0, cost_limit=1.0, lambda_lr=0.5)
        try:
            module = algo.learner_group._learner.module[DEFAULT_MODULE_ID].unwrapped()
            initial_lambda = float(module.get_lambda().detach().cpu().numpy())

            results = algo.train()
            check_train_results_new_api_stack(results)

            stats = results[LEARNER_RESULTS][DEFAULT_MODULE_ID]
            self.assertIn("mean_cost", stats)
            self.assertIn("lambda_value", stats)

            # Cost should come from env infos["cost"]. The extra bootstrap timestep
            # may introduce one default 0.0, so only assert it's close enough.
            self.assertGreater(stats["mean_cost"], 8.0)

            new_lambda = float(module.get_lambda().detach().cpu().numpy())
            self.assertGreater(new_lambda, initial_lambda)
        finally:
            algo.stop()

    def test_train_decreases_lambda_when_cost_below_limit(self):
        algo = self._build_algo(cost=0.0, cost_limit=5.0, lambda_lr=0.5)
        try:
            module = algo.learner_group._learner.module[DEFAULT_MODULE_ID].unwrapped()
            initial_lambda = float(module.get_lambda().detach().cpu().numpy())

            results = algo.train()
            check_train_results_new_api_stack(results)

            stats = results[LEARNER_RESULTS][DEFAULT_MODULE_ID]
            self.assertIn("mean_cost", stats)
            self.assertIn("lambda_value", stats)
            self.assertLess(stats["mean_cost"], 1.0)

            new_lambda = float(module.get_lambda().detach().cpu().numpy())
            self.assertLess(new_lambda, initial_lambda)
        finally:
            algo.stop()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))
