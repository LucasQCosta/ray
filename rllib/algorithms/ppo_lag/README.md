# PPO-Lagrangian (PPO-Lag)

## Overview

PPO-Lag (a.k.a. *PPO-Lagrangian*) extends
[PPO](https://arxiv.org/abs/1707.06347) to **constrained reinforcement learning**.

Instead of optimizing reward only, PPO-Lag optimizes reward **subject to a cost
constraint**, using a primal-dual method with a Lagrange multiplier $\lambda$.

In RLlib, the environment provides the per-step cost via `info["cost"]` (float). RLlib
then aggregates this into episode/batch cost statistics (e.g., `mean_cost`).

For a recent discussion of Lagrangian (primal-dual) constrained policy optimization
in the context of PPO-style methods, see: https://arxiv.org/html/2510.17564v1

## Constrained objective (primal-dual)

We want to solve:

$$
\max_{\theta} \; J_R(\theta) \quad\text{s.t.}\quad J_C(\theta) \le d
$$

where:
- $J_R(\theta)$ is the expected return (reward)
- $J_C(\theta)$ is the expected cost return (sum of costs)
- $d$ is the cost limit (`cost_limit`)

The Lagrangian is:

$$
\mathcal{L}(\theta, \lambda) = J_R(\theta) - \lambda\,(J_C(\theta) - d), \quad \lambda \ge 0
$$

Training alternates:
- **Primal step (policy/value update)**: maximize $\mathcal{L}$ w.r.t. $\theta$
- **Dual step (multiplier update)**: increase $\lambda$ if the observed cost exceeds $d$

A typical projected dual update is:

$$
\lambda \leftarrow \max\{0,\; \lambda + \alpha_\lambda\,(\hat{J}_C - d)\}
$$

where $\alpha_\lambda$ is `lambda_lr` and $\hat{J}_C$ is the cost estimate from the
current batch.

## What changes vs PPO (clipped surrogate with cost)

PPO uses the clipped surrogate objective:

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

$$
\mathcal{L}^{CLIP}(\theta)=\mathbb{E}_t\Big[\min\big(r_t\,\hat{A}_t,\; \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\,\hat{A}_t\big)\Big]
$$

In PPO-Lag, the advantage is replaced by a **Lagrangian advantage**:

$$
\hat{A}^{Lag}_t = \hat{A}^R_t - \lambda\,\hat{A}^C_t
$$

Intuitively: it is “standard PPO”, but it trades off reward improvement against cost
reduction via $\lambda$.

Implementation-wise this typically adds:
- a **cost value function** (cost critic) and its loss
- computation of **cost advantages** (GAE-style) from per-step costs
- a **dual update for $\lambda$** (and related logging)

## Added functionality vs PPO (as implemented here)

Compared to unconstrained PPO, this PPO-Lagrangian implementation adds the following
building blocks (mirroring the standard primal-dual Lagrangian recipe; see
https://arxiv.org/html/2510.17564v1):

- **Cost signal plumbing**: expects the environment to emit `info["cost"]` each step.
- **Constraint knob**: `cost_limit = d` sets the target upper bound for expected cost.
- **Dual variable**: maintains a non-negative multiplier $\lambda$ and updates it when
	observed cost exceeds the limit.
- **Cost critic & advantages**: estimates cost value/advantages (analogous to reward
	value/GAE) and uses them inside the Lagrangian advantage.
- **Extra metrics**: logs cost statistics and $\lambda$ (e.g., `mean_cost`, `lambda_value`).

## RLlib implementation map (this folder)

New API Stack (RLModule + Learner):
- [ppo_lag.py](ppo_lag.py): defines `PPOConfig`/`PPO` for PPO-Lag.
- [ppo_lag_learner.py](ppo_lag_learner.py): learner logic (losses, stats, $\lambda$ dual update).
- [ppo_lag_rl_module.py](ppo_lag_rl_module.py): RLModule for PPO-Lag.
- [ppo_lag_catalog.py](ppo_lag_catalog.py): component/model construction.
- [default_ppo_lag_rl_module.py](default_ppo_lag_rl_module.py): default RLModule.

Torch-specific pieces:
- [torch/ppo_lag_torch_learner.py](torch/ppo_lag_torch_learner.py)
- [torch/ppo_lag_torch_rl_module.py](torch/ppo_lag_torch_rl_module.py)

Old API Stack compatibility:
- [ppo_lag_tf_policy.py](ppo_lag_tf_policy.py)
- [ppo_lag_torch_policy.py](ppo_lag_torch_policy.py)

If you are using the new stack (e.g. `.api_stack(enable_rl_module_and_learner=True, ...)`),
training/inference goes through `RLModule`/`Learner`. The `*_policy.py` files are for
the legacy Policy-based stack.

## Environment requirements (cost signal)

Your environment should emit a scalar cost each step:

```python
obs, reward, terminated, truncated, info = env.step(action)
info["cost"] = float(cost_t)
```

Episode cost is typically the sum over steps.

## Usage

```python
from ray.rllib.algorithms import ppo_lag

config = (
	ppo_lag.PPOConfig()
	.environment("my_env", env_config={...})
	.framework("torch")
	.api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
)

config.cost_limit = 25.0
config.lambda_lr = 0.01

algo = config.build()
for _ in range(100):
	result = algo.train()
```

## References

- PPO: https://arxiv.org/abs/1707.06347
- PPO-Lagrangian reference: https://arxiv.org/html/2510.17564v1
