import unittest

import gymnasium as gym
import numpy as np

import ray
import ray.rllib.algorithms.ppo_lag as ppo_lag
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch


class TestPPOLagLearner(unittest.TestCase):
    ENV = gym.make("CartPole-v1")

    @classmethod
    def setUpClass(cls):
        ray.init()

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def _build_local_learner(self, *, add_default_connectors: bool):
        config = (
            ppo_lag.PPOConfig()
            .framework("torch")
            .environment("CartPole-v1")
            .env_runners(num_env_runners=0)
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .learners(
                num_learners=0,
                add_default_connectors_to_learner_pipeline=add_default_connectors,
            )
            .training(
                # Keep the learner update fast.
                num_epochs=1,
                minibatch_size=None,
                train_batch_size=32,
                use_kl_loss=False,
            )
        )
        algo_config = config.copy(copy_frozen=False)
        algo_config.validate()
        algo_config.freeze()
        learner_group = algo_config.build_learner_group(env=self.ENV)
        return learner_group._learner

    def _make_synthetic_mab(self, *, t: int, mean_cost: float) -> MultiAgentBatch:
        obs = np.random.uniform(low=-1.0, high=1.0, size=(t, 4)).astype(np.float32)
        next_obs = np.random.uniform(low=-1.0, high=1.0, size=(t, 4)).astype(
            np.float32
        )
        actions = np.random.randint(low=0, high=2, size=(t,), dtype=np.int64)
        rewards = np.random.uniform(low=-1.0, high=1.0, size=(t,)).astype(np.float32)
        terminateds = np.zeros((t,), dtype=bool)
        terminateds[-1] = True
        truncateds = np.zeros((t,), dtype=bool)

        action_dist_inputs = np.random.normal(size=(t, 2)).astype(np.float32)
        action_logp = np.random.normal(loc=-0.5, scale=0.1, size=(t,)).astype(
            np.float32
        )

        advantages = np.random.normal(size=(t,)).astype(np.float32)
        value_targets = np.random.normal(size=(t,)).astype(np.float32)

        costs = (mean_cost * np.ones((t,), dtype=np.float32))
        cost_advantages = np.random.normal(size=(t,)).astype(np.float32)
        cost_value_targets = np.random.normal(size=(t,)).astype(np.float32)

        batch = SampleBatch(
            {
                Columns.OBS: obs,
                Columns.NEXT_OBS: next_obs,
                Columns.ACTIONS: actions,
                Columns.REWARDS: rewards,
                Columns.TERMINATEDS: terminateds,
                Columns.TRUNCATEDS: truncateds,
                Columns.ACTION_DIST_INPUTS: action_dist_inputs,
                Columns.ACTION_LOGP: action_logp,
                Postprocessing.ADVANTAGES: advantages,
                Postprocessing.VALUE_TARGETS: value_targets,
                Columns.EPS_ID: np.zeros((t,), dtype=np.int64),
                "costs": costs,
                "cost_advantages": cost_advantages,
                "cost_value_targets": cost_value_targets,
            }
        )

        return MultiAgentBatch({DEFAULT_MODULE_ID: batch}, env_steps=t)

    def test_build_adds_cost_gae_connector(self):
        learner = self._build_local_learner(add_default_connectors=True)
        connector_names = [type(c).__name__ for c in learner._learner_connector.connectors]

        self.assertIn("AddOneTsToEpisodesAndTruncate", connector_names)
        self.assertIn("AddInfosFromEpisodesToTrainBatch", connector_names)
        self.assertIn("AddCostsFromInfos", connector_names)
        self.assertIn("GeneralAdvantageEstimation", connector_names)
        self.assertIn("CostGeneralAdvantageEstimation", connector_names)

        # Need infos before we can derive costs from them.
        self.assertLess(
            connector_names.index("AddInfosFromEpisodesToTrainBatch"),
            connector_names.index("AddCostsFromInfos"),
        )

        # Cost GAE should come after reward GAE in the pipeline.
        self.assertGreater(
            connector_names.index("CostGeneralAdvantageEstimation"),
            connector_names.index("GeneralAdvantageEstimation"),
        )

        # Costs must be derived before cost GAE runs.
        self.assertLess(
            connector_names.index("AddCostsFromInfos"),
            connector_names.index("CostGeneralAdvantageEstimation"),
        )

    def test_lambda_updates_and_metrics_logged(self):
        learner = self._build_local_learner(add_default_connectors=False)

        module = learner.module[DEFAULT_MODULE_ID].unwrapped()
        initial_lambda = float(module.get_lambda().detach().cpu().numpy())

        mab = self._make_synthetic_mab(t=16, mean_cost=10.0)

        # Make lambda update more visible.
        learner.config.cost_limit = 5.0
        learner.config.lambda_lr = 0.5

        results = learner.update(batch=mab, num_epochs=1)

        # Metrics must be present.
        self.assertIn(DEFAULT_MODULE_ID, results)
        self.assertIn("mean_cost", results[DEFAULT_MODULE_ID])
        self.assertIn("lambda_value", results[DEFAULT_MODULE_ID])

        # Lambda should increase because mean_cost > cost_limit.
        new_lambda = float(module.get_lambda().detach().cpu().numpy())
        self.assertGreater(new_lambda, initial_lambda)

    def test_lambda_decreases_when_below_cost_limit(self):
        learner = self._build_local_learner(add_default_connectors=False)
        module = learner.module[DEFAULT_MODULE_ID].unwrapped()

        initial_lambda = float(module.get_lambda().detach().cpu().numpy())

        learner.config.cost_limit = 5.0
        learner.config.lambda_lr = 1.0

        mab = self._make_synthetic_mab(t=16, mean_cost=0.0)
        learner.update(batch=mab, num_epochs=1)

        new_lambda = float(module.get_lambda().detach().cpu().numpy())
        self.assertLess(new_lambda, initial_lambda)

    def test_lambda_optimizer_is_separate(self):
        learner = self._build_local_learner(add_default_connectors=False)
        module = learner.module[DEFAULT_MODULE_ID].unwrapped()

        optimizers = dict(learner.get_optimizers_for_module(DEFAULT_MODULE_ID))
        self.assertIn("default_optimizer", optimizers)
        self.assertIn("lambda_optimizer", optimizers)

        default_optim = optimizers["default_optimizer"]
        lambda_optim = optimizers["lambda_optimizer"]

        default_params = {
            p
            for g in default_optim.param_groups
            for p in g.get("params", [])
        }
        lambda_params = {
            p
            for g in lambda_optim.param_groups
            for p in g.get("params", [])
        }

        self.assertNotIn(module.lambda_param, default_params)
        self.assertEqual(lambda_params, {module.lambda_param})

    def test_state_restore_preserves_lambda(self):
        learner1 = self._build_local_learner(add_default_connectors=False)
        module1 = learner1.module[DEFAULT_MODULE_ID].unwrapped()

        learner1.config.cost_limit = 5.0
        learner1.config.lambda_lr = 0.5

        mab = self._make_synthetic_mab(t=16, mean_cost=10.0)
        learner1.update(batch=mab, num_epochs=1)

        lambda_after_update = float(module1.get_lambda().detach().cpu().numpy())
        state = learner1.get_state()

        learner2 = self._build_local_learner(add_default_connectors=False)
        module2 = learner2.module[DEFAULT_MODULE_ID].unwrapped()
        learner2.set_state(state)

        lambda_restored = float(module2.get_lambda().detach().cpu().numpy())
        self.assertAlmostEqual(lambda_restored, lambda_after_update, places=6)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))
