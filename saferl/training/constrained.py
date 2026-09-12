"""Constrained PPO: the learned soft layer of the phase-1 hybrid design.

The hard shield (`saferl.shield`) guarantees safety by overriding unsafe
actions. It does not, on its own, teach the policy anything -- an agent can
ride the shield forever and never learn to stay away from hazards. This
module supplies the pressure that makes the policy internalise the shield's
boundary: every shield intervention costs it reward.

**What this is, precisely.** The Lagrange multiplier is updated by a genuine
dual-gradient ascent step on the constraint violation:

    lambda <- clip(lambda + lr * (intervention_rate - target_rate), 0, max)

so lambda rises while the policy exceeds its intervention budget and decays
once it is under. That part is the real thing, not a schedule.

**What it is not.** The multiplier is applied by *shaping the scalar reward*
(`r' = r - lambda * cost`) and learned through PPO's single existing critic.
A textbook PPO-Lagrangian trains a separate cost-value head and forms the
policy gradient from both critics. This is the simpler construction -- the
same dual variable, one critic instead of two. It is called out here rather
than dressed up, and the README says the same.
"""
import numpy as np
import gymnasium as gym


class CostPenaltyWrapper(gym.Wrapper):
    """Subtracts `lambda * cost` from reward and adapts lambda to a budget.

    Reads `info["shield_cost"]` (1 on any step the shield intervened) from
    `ShieldedEnv`. With `enabled=False` the multiplier is pinned at zero, so
    the unconstrained baseline runs through byte-identical plumbing and the
    comparison isn't confounded by a different code path.
    """

    def __init__(self, env, target_rate=0.002, lambda_lr=2.0, lambda_init=0.0,
                 lambda_max=50.0, ema_beta=0.9, enabled=True,
                 cost_key="shield_cost"):
        super().__init__(env)
        self.target_rate = target_rate
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        self.ema_beta = ema_beta
        self.enabled = enabled
        self.cost_key = cost_key

        self.lam = float(lambda_init) if enabled else 0.0
        self.ema_rate = None          # EMA of per-episode intervention rate
        self._ep_cost = 0.0
        self._ep_steps = 0
        self.lambda_history = []

    def reset(self, **kwargs):
        self._ep_cost = 0.0
        self._ep_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        cost = float(info.get(self.cost_key, 0.0))
        penalty = self.lam * cost
        self._ep_cost += cost
        self._ep_steps += 1

        info["task_reward"] = float(reward)      # before the penalty
        info["penalty"] = float(penalty)
        info["lambda"] = float(self.lam)

        if done or truncated:
            self._update_lambda()

        return obs, float(reward) - penalty, done, truncated, info

    def _update_lambda(self):
        """One dual-gradient step, taken on episode boundaries.

        The constraint is on the *rate* rather than the episode total, because
        the out-of-bounds backstop makes episode length vary by ~4x and a
        per-episode budget would just reward ending episodes early.
        """
        if self._ep_steps == 0:
            return
        rate = self._ep_cost / self._ep_steps
        self.ema_rate = rate if self.ema_rate is None else (
            self.ema_beta * self.ema_rate + (1.0 - self.ema_beta) * rate)
        if self.enabled:
            self.lam = float(np.clip(
                self.lam + self.lambda_lr * (self.ema_rate - self.target_rate),
                0.0, self.lambda_max))
        self.lambda_history.append(self.lam)
