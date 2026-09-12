"""Analytic hard-constraint safety layer.

Phase 5 replaced the phase-1 placeholder. The shield now:

  * judges an action by where it *leads*, using relative position AND
    relative velocity over a short lookahead, rather than by the agent's
    current distance to a hazard;
  * on a trigger, substitutes the **least-restrictive safe action** -- the
    safe action whose commanded thrust is closest to what the policy asked
    for -- rather than a uniformly random one;
  * falls back to the action that delays the predicted breach longest when
    no action is safe, and records that case separately.

`RandomReplacementShield` preserves the old placeholder behaviour. It is not
wired into training; it exists so `saferl.eval.stress_test` can measure the
new shield against the old one on identical scenarios.
"""
from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from saferl.env.base_env import (
    ACTION_THRUST_DIRS,
    AGENT_MASS,
    OBS_HEADER_LEN,
    OBS_PER_HAZARD,
    PHYSICS_HZ,
)

# Defaults matching saferl/configs/default.yaml's env block, so a bare
# SafetyShield() still models the default environment correctly.
DEFAULT_ACCEL = 12.0 / AGENT_MASS          # force_mag / mass  = 1.2 m/s^2
DEFAULT_STEP_DT = 10 / PHYSICS_HZ          # sim_substeps / physics rate


@dataclass
class ShieldDecision:
    """Structured record of one shield evaluation.

    Kept per-step (and optionally appended to `SafetyShield.log`) because an
    integer counter throws away everything useful for later evaluation: which
    hazard triggered, how close the executed action is predicted to get, and
    whether a safe alternative existed at all.
    """
    kind: str                  # "none" | "substituted" | "fallback"
    proposed_action: int
    executed_action: int
    intervened: bool
    hazard_index: int = -1     # hazard that triggered the check (-1 if none)
    predicted_min_dist: float = float("inf")   # for the executed action
    time_to_violation: float = float("inf")    # for the executed action, sec
    deviation: float = 0.0     # |thrust(executed) - thrust(proposed)|
    n_safe_actions: int = 0


class SafetyShield:
    """Least-restrictive safe-action shield over the Discrete(4) action set.

    Model used for the lookahead (all deliberately simple, and all stated
    here so the approximations are auditable):

      * The agent is a point mass under constant thrust for the whole
        horizon:  p(t) = p0 + v0*t + 0.5*a*t^2, with `a` read from
        `ACTION_THRUST_DIRS` so it cannot drift out of sync with the env.
      * Debris move at constant velocity: h(t) = h0 + hv*t. Boundary
        reflection is ignored -- a bounce can only move a hazard *away* from
        a straight-line prediction near the wall, so ignoring it is the
        conservative direction.
      * The z axis is dropped from the velocity propagation. Thrust is purely
        in x/y and the agent rests on the plane, so its z velocity is contact
        settling, not commandable motion; propagating it would let the shield
        believe the agent can duck under a hazard. The z *offset* between
        agent and hazard is kept, so the separation stays a true 3D distance.
    """

    def __init__(self, safe_dist=2.2, lookahead_steps=40,
                 accel=DEFAULT_ACCEL, step_dt=DEFAULT_STEP_DT,
                 keep_log=False, log_limit=10000):
        self.safe_dist = safe_dist
        self.lookahead_steps = int(lookahead_steps)
        self.accel = float(accel)
        self.step_dt = float(step_dt)

        # thrust acceleration vector per action, (A, 3)
        self._acc = ACTION_THRUST_DIRS * self.accel
        # sample times, excluding t=0: the question this shield answers is
        # "does this action lead into a hazard", not "am I near one already".
        self._ts = self.step_dt * np.arange(1, self.lookahead_steps + 1)
        self._ts2 = self._ts ** 2

        # aggregate counters -- cumulative over the shield's lifetime
        self.n_checks = 0
        self.n_triggered = 0
        self.n_substituted = 0
        self.n_fallback = 0

        self.keep_log = keep_log
        self.log_limit = log_limit
        self.log = []
        self.last_decision = None

    # ------ observation decoding ------
    def _hazards(self, obs):
        """Hazard (positions, velocities) as (H, 3) arrays.

        Skips zero-padded blocks (the curriculum pads the obs with zeros when
        fewer than max_hazards are live) and any trailing partial block, so a
        short hand-built obs in a test does not index off the end.
        """
        pos, vel = [], []
        for i in range(OBS_HEADER_LEN, len(obs), OBS_PER_HAZARD):
            if i + OBS_PER_HAZARD > len(obs):
                break
            h_pos = np.asarray(obs[i:i + 3], dtype=np.float64)
            if np.all(h_pos == 0):
                continue
            pos.append(h_pos)
            vel.append(np.asarray(obs[i + 3:i + 6], dtype=np.float64))
        if not pos:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.stack(pos), np.stack(vel)

    # ------ core prediction ------
    def _predict(self, obs):
        """Predicted separation for every (action, hazard) pair.

        Returns (min_dist, t_violation), each shaped (num_actions, H):
        the closest the pair is predicted to come within the horizon, and the
        first sample time at which the pair is predicted inside safe_dist
        (inf if it never is).
        """
        p0 = np.asarray(obs[0:3], dtype=np.float64)
        v0 = np.asarray(obs[3:6], dtype=np.float64).copy()
        v0[2] = 0.0  # see class docstring: z velocity is settling, not control

        h_pos, h_vel = self._hazards(obs)
        n_h = h_pos.shape[0]
        n_a = self._acc.shape[0]
        if n_h == 0:
            inf = np.full((n_a, 0), np.inf)
            return inf, inf.copy()

        r0 = p0[None, :] - h_pos            # (H, 3) relative position
        dv = v0[None, :] - h_vel            # (H, 3) relative velocity
        ts = self._ts

        # (A, H, K, 3): action a, hazard h, lookahead sample k
        disp = (r0[None, :, None, :]
                + dv[None, :, None, :] * ts[None, None, :, None]
                + 0.5 * self._acc[:, None, None, :] * self._ts2[None, None, :, None])
        dist = np.linalg.norm(disp, axis=-1)      # (A, H, K)

        min_dist = dist.min(axis=2)               # (A, H)
        breach = dist < self.safe_dist
        any_breach = breach.any(axis=2)
        first_k = breach.argmax(axis=2)           # first True index per (A, H)
        t_violation = np.where(any_breach, ts[first_k], np.inf)
        return min_dist, t_violation

    def _deviation(self, action, proposed):
        """How far a candidate action is from what the policy asked for.

        Euclidean distance between the two commanded thrust vectors. Action
        *indices* are meaningless as a metric -- 0 and 1 are adjacent indices
        but opposite directions -- whereas thrust distance is physical: 0 for
        the same action, |a|*sqrt(2) for a perpendicular sidestep, 2|a| for a
        full reversal. So a sidestep is always preferred over a reversal.
        """
        return float(np.linalg.norm(self._acc[action] - self._acc[proposed]))

    # ------ public API ------
    def check_and_fix(self, obs, action):
        """Return (safe_action, intervened).

        The 2-tuple contract is unchanged from phase 1 so ShieldedEnv and any
        existing caller keep working; the richer signal lands on
        `self.last_decision` and the aggregate counters.
        """
        proposed = int(action)
        self.n_checks += 1

        min_dist, t_violation = self._predict(obs)
        if min_dist.shape[1] == 0:               # no live hazards
            return self._record(ShieldDecision("none", proposed, proposed, False))

        # worst case over hazards, per action
        action_min = min_dist.min(axis=1)                  # (A,)
        action_tviol = t_violation.min(axis=1)             # (A,)
        safe = ~np.isfinite(action_tviol)

        # the hazard that triggered: earliest predicted breach of the
        # proposed action, falling back to whichever it comes closest to
        if np.isfinite(action_tviol[proposed]):
            trigger = int(np.argmin(t_violation[proposed]))
        else:
            trigger = int(np.argmin(min_dist[proposed]))

        if safe[proposed]:
            return self._record(ShieldDecision(
                "none", proposed, proposed, False,
                hazard_index=trigger,
                predicted_min_dist=float(action_min[proposed]),
                time_to_violation=float(action_tviol[proposed]),
                n_safe_actions=int(safe.sum()),
            ))

        self.n_triggered += 1
        candidates = np.flatnonzero(safe)

        if candidates.size:
            # least-restrictive: smallest thrust deviation from the policy's
            # intent; ties (a left vs right sidestep are equidistant) broken
            # toward the larger predicted clearance.
            best = min(candidates,
                       key=lambda a: (self._deviation(a, proposed), -action_min[a]))
            kind = "substituted"
            self.n_substituted += 1
        else:
            # boxed in: nothing keeps safe_dist over the horizon. Buy time --
            # maximise time-to-violation, tie-break on clearance.
            best = max(range(self._acc.shape[0]),
                       key=lambda a: (action_tviol[a], action_min[a]))
            kind = "fallback"
            self.n_fallback += 1

        best = int(best)
        return self._record(ShieldDecision(
            kind, proposed, best, True,
            hazard_index=trigger,
            predicted_min_dist=float(action_min[best]),
            time_to_violation=float(action_tviol[best]),
            deviation=self._deviation(best, proposed),
            n_safe_actions=int(candidates.size),
        ))

    def is_action_safe(self, obs, action):
        """Whether `action` passes the lookahead check from `obs`.

        Exposed so evaluation code can audit *any* shield's chosen action
        (including the random baseline's) against one consistent predicate.
        """
        min_dist, _ = self._predict(obs)
        if min_dist.shape[1] == 0:
            return True
        return bool(min_dist[int(action)].min() >= self.safe_dist)

    def safe_actions(self, obs):
        """Indices of every action that passes the lookahead check."""
        min_dist, _ = self._predict(obs)
        if min_dist.shape[1] == 0:
            return list(range(self._acc.shape[0]))
        return [a for a in range(self._acc.shape[0])
                if min_dist[a].min() >= self.safe_dist]

    def _record(self, decision):
        self.last_decision = decision
        if self.keep_log and len(self.log) < self.log_limit:
            self.log.append(decision)
        return decision.executed_action, decision.intervened


class RandomReplacementShield:
    """The phase-1 placeholder, kept only as a measurement baseline.

    Triggers on current distance alone and replaces the action with a uniform
    random draw -- which, with moving debris, can itself steer into a hazard.
    `saferl.eval.stress_test` runs this against SafetyShield to put a number
    on that failure mode. Not used in training.
    """

    def __init__(self, safe_dist=2.2, rng=None):
        self.safe_dist = safe_dist
        self.rng = rng if rng is not None else np.random
        self.n_checks = 0
        self.n_triggered = 0
        self.n_substituted = 0
        self.n_fallback = 0          # never fires; kept for a uniform interface
        self.last_decision = None

    def check_and_fix(self, obs, action):
        self.n_checks += 1
        pos = np.asarray(obs[0:3], dtype=np.float64)
        for idx, i in enumerate(range(OBS_HEADER_LEN, len(obs), OBS_PER_HAZARD)):
            h_pos = np.asarray(obs[i:i + 3], dtype=np.float64)
            if h_pos.shape[0] < 3 or np.all(h_pos == 0):
                continue
            if np.linalg.norm(pos - h_pos) < self.safe_dist:
                replacement = int(self.rng.choice([0, 1, 2, 3]))
                self.n_triggered += 1
                self.n_substituted += 1
                self.last_decision = ShieldDecision(
                    "substituted", int(action), replacement, True, hazard_index=idx,
                )
                return replacement, True
        self.last_decision = ShieldDecision("none", int(action), int(action), False)
        return action, False


class ShieldedEnv(gym.Wrapper):
    """Calls the shield, counts interventions, passes everything else through.

    Phase 5 added `fallback_interventions` only. The wrapper's role is
    unchanged: it does not make safety decisions, it just records what the
    shield decided, and the boxed-in case is a materially different event
    from a normal substitution -- collapsing both into one counter would hide
    exactly the signal phase 9's evaluation needs.
    """

    def __init__(self, env, shield):
        super().__init__(env)
        self.shield = shield
        self.interventions = 0
        self.fallback_interventions = 0
        self._last_obs = None  # obs tracked here, not via a private env method

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        safe_action, intervened = self.shield.check_and_fix(self._last_obs, action)
        if intervened:
            self.interventions += 1
            decision = getattr(self.shield, "last_decision", None)
            if decision is not None and decision.kind == "fallback":
                self.fallback_interventions += 1
        obs, reward, done, truncated, info = self.env.step(safe_action)
        self._last_obs = obs
        return obs, reward, done, truncated, info
