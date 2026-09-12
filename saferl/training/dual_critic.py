"""Two-critic PPO-Lagrangian for constrained RL.

Phase 6b: the single-critic approach from phase 6 collapsed to 0% goal rate
because one value function cannot distinguish "I earned task reward" from
"I avoided cost". This module adds a separate cost-value head so the policy
gradient can credit-assign reward and cost independently.

Architecture: ``DualCriticPolicy`` subclasses SB3's ``ActorCriticPolicy``
with a second ``cost_value_net``. ``ConstrainedPPO`` subclasses ``PPO`` to
collect cost signals into a ``CostRolloutBuffer``, compute cost-specific GAE
advantages, and form the combined policy gradient
``A_task - lambda * A_cost``. Lambda is updated by dual-gradient ascent
on the observed intervention rate against a curriculum-scheduled target.
"""
from __future__ import annotations

from collections import namedtuple
from functools import partial

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor

CostRolloutBufferSamples = namedtuple(
    "CostRolloutBufferSamples",
    RolloutBufferSamples._fields + ("cost_returns", "cost_old_values"),
)


class CostRolloutBuffer(RolloutBuffer):
    """RolloutBuffer extended with parallel cost-signal tracking."""

    def reset(self):
        super().reset()
        self.cost_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.task_advantages_raw = None

    def add_cost(self, cost_reward: np.ndarray, cost_value: th.Tensor):
        pos = self.pos - 1
        if pos < 0:
            pos = self.buffer_size - 1
        self.cost_rewards[pos] = np.array(cost_reward)
        self.cost_values[pos] = cost_value.clone().cpu().numpy().flatten()

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray,
        last_cost_values: th.Tensor | None = None,
    ) -> None:
        super().compute_returns_and_advantage(last_values, dones)
        self.task_advantages_raw = self.advantages.copy()

        if last_cost_values is not None:
            last_cv = last_cost_values.clone().cpu().numpy().flatten()
            last_gae = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_nt = 1.0 - dones.astype(np.float32)
                    next_vals = last_cv
                else:
                    next_nt = 1.0 - self.episode_starts[step + 1]
                    next_vals = self.cost_values[step + 1]
                delta = (self.cost_rewards[step]
                         + self.gamma * next_vals * next_nt
                         - self.cost_values[step])
                last_gae = delta + self.gamma * self.gae_lambda * next_nt * last_gae
                self.cost_advantages[step] = last_gae
            self.cost_returns = self.cost_advantages + self.cost_values

    def set_combined_advantages(self, lam: float):
        if self.task_advantages_raw is not None:
            self.advantages = self.task_advantages_raw - lam * self.cost_advantages

    def get(self, batch_size=None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for name in ("observations", "actions", "values", "log_probs",
                         "advantages", "returns", "cost_returns", "cost_values"):
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        start = 0
        while start < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start:start + batch_size])
            start += batch_size

    def _get_samples(self, batch_inds, env=None):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
            self.cost_values[batch_inds].flatten(),
        )
        return CostRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class DualCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with an additional cost-value head."""

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.cost_value_net = th.nn.Linear(
            self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            self.cost_value_net.apply(partial(self.init_weights, gain=1))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def predict_cost_values(self, obs):
        features = self.extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.cost_value_net(latent_vf)

    def forward_all(self, obs, actions):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_vf)
        entropy = distribution.entropy()
        return values, cost_values, log_prob, entropy


class ConstrainedPPO(PPO):
    """PPO with dual-gradient Lagrangian constraint on shield interventions.

    The combined advantage ``A_task - lam * A_cost`` goes into the clipped
    surrogate objective. ``lam`` is updated by dual-gradient ascent on the
    observed intervention rate against a curriculum-scheduled target that
    ramps from ``target_rate_start`` (loose) to ``target_rate_end`` (tight)
    over training.
    """

    def __init__(self, policy, env, *,
                 target_rate_start: float = 0.25,
                 target_rate_end: float = 0.05,
                 constraint_lambda_lr: float = 0.1,
                 constraint_lambda_max: float = 5.0,
                 constraint_ema_beta: float = 0.9,
                 cost_vf_coef: float = 0.5,
                 **kwargs):
        self.target_rate_start = target_rate_start
        self.target_rate_end = target_rate_end
        self.constraint_lambda_lr = constraint_lambda_lr
        self.constraint_lambda_max = constraint_lambda_max
        self.constraint_ema_beta = constraint_ema_beta
        self.cost_vf_coef = cost_vf_coef
        self.lam = 0.0
        self.ema_rate = None
        self.lambda_history: list[float] = []
        self.target_rate_history: list[float] = []
        kwargs.setdefault("rollout_buffer_class", CostRolloutBuffer)
        super().__init__(policy, env, **kwargs)

    @property
    def current_target_rate(self) -> float:
        p = self._current_progress_remaining
        return self.target_rate_end + p * (self.target_rate_start - self.target_rate_end)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()

        ep_costs: list[float] = []
        _cur_cost, _cur_steps = 0.0, 0

        while n_steps < n_rollout_steps:
            if (self.use_sde and self.sde_sample_freq > 0
                    and n_steps % self.sde_sample_freq == 0):
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_t = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_t)
                cost_values = self.policy.predict_cost_values(obs_t)

            actions_np = actions.cpu().numpy()
            clipped = actions_np
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped = self.policy.unscale_action(clipped)
                else:
                    clipped = np.clip(actions_np,
                                      self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped)
            cost_reward = np.array(
                [float(info.get("shield_cost", 0.0)) for info in infos])

            _cur_cost += float(cost_reward[0])
            _cur_steps += 1
            if dones[0]:
                if _cur_steps > 0:
                    ep_costs.append(_cur_cost / _cur_steps)
                _cur_cost, _cur_steps = 0.0, 0

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        tv = self.policy.predict_values(terminal_obs)[0]
                        tcv = self.policy.predict_cost_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * tv
                    cost_reward[idx] += self.gamma * float(tcv)

            rollout_buffer.add(self._last_obs, actions_np, rewards,
                               self._last_episode_starts, values, log_probs)
            rollout_buffer.add_cost(cost_reward, cost_values)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            obs_t = obs_as_tensor(new_obs, self.device)
            last_values = self.policy.predict_values(obs_t)
            last_cost_values = self.policy.predict_cost_values(obs_t)

        rollout_buffer.compute_returns_and_advantage(
            last_values, dones, last_cost_values=last_cost_values)

        if ep_costs:
            mean_rate = float(np.mean(ep_costs))
            target = self.current_target_rate
            self.ema_rate = mean_rate if self.ema_rate is None else (
                self.constraint_ema_beta * self.ema_rate
                + (1.0 - self.constraint_ema_beta) * mean_rate)
            self.lam = float(np.clip(
                self.lam + self.constraint_lambda_lr * (self.ema_rate - target),
                0.0, self.constraint_lambda_max))
            self.lambda_history.append(self.lam)
            self.target_rate_history.append(target)

        rollout_buffer.set_combined_advantages(self.lam)
        callback.on_rollout_end()
        return True

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (self.clip_range_vf(self._current_progress_remaining)
                         if self.clip_range_vf is not None else None)

        ent_losses, pg_losses, vf_losses, cvf_losses = [], [], [], []
        clip_fractions = []
        cost_estimates, cost_observed = [], []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, cost_values, log_prob, entropy = \
                    self.policy.forward_all(rollout_data.observations, actions)
                values = values.flatten()
                cost_values = cost_values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = ((advantages - advantages.mean())
                                  / (advantages.std() + 1e-8))

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                pl1 = advantages * ratio
                pl2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(pl1, pl2).mean()
                pg_losses.append(policy_loss.item())

                clip_frac = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_frac)

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                vf_losses.append(value_loss.item())

                cost_value_loss = F.mse_loss(
                    rollout_data.cost_returns, cost_values)
                cvf_losses.append(cost_value_loss.item())
                # Logged as two separate traces rather than only their MSE:
                # the loss alone cannot distinguish a critic that is
                # well-calibrated but noisy from one that is systematically
                # over- or under-predicting cost, and that bias is the thing
                # that decides whether lambda is pushing on a real signal.
                cost_estimates.append(cost_values.mean().item())
                cost_observed.append(rollout_data.cost_returns.mean().item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                ent_losses.append(entropy_loss.item())

                loss = (policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.cost_vf_coef * cost_value_loss)

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl)

                if (self.target_kl is not None
                        and approx_kl > 1.5 * self.target_kl):
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        from stable_baselines3.common.utils import explained_variance
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(ent_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(vf_losses))
        self.logger.record("train/cost_value_loss", np.mean(cvf_losses))
        self.logger.record("train/cost_value_estimate", np.mean(cost_estimates))
        self.logger.record("train/cost_return_observed", np.mean(cost_observed))
        self.logger.record("train/cost_value_bias",
                           np.mean(cost_estimates) - np.mean(cost_observed))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/lambda", self.lam)
        self.logger.record("train/target_rate", self.current_target_rate)
        if self.ema_rate is not None:
            self.logger.record("train/ema_intervention_rate", self.ema_rate)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
