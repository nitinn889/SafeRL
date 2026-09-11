"""
shielded_agent.py
=================
Integrates a trained RL agent (PPO or SAC) with the probabilistic shield.

The ShieldedAgent wraps any base agent and intercepts its action predictions,
applying the PRISM-based safety shield before returning the final action to
the environment. This produces the full SafeRL pipeline:

    observation → RL policy → proposed action
                           ↘
                         Shield (PRISM PMC)
                           ↘
                         [safe?] → yes → execute
                                 → no  → substitute least-restrictive safe action

The shielded agent is evaluated against:
  - Unshielded PPO / SAC: measures the cost (collision rate) improvement.
  - Shield-only (myopic safe policy): measures the task-success improvement.

Usage
-----
>>> from agents.shielded_agent import ShieldedAgent
>>> from shield.shield_query import PRISMShieldQuery
>>> agent = ShieldedAgent(
...     base_agent=trained_ppo,
...     shield_query=PRISMShieldQuery.from_table("shield/prism_models/p_collision_table.npy"),
...     safety_threshold=0.05,
... )
>>> action = agent.predict(obs)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .ppo_agent import PPOAgent
from .sac_agent import SACAgent
from shield.abstraction import StateAbstraction
from shield.shield_query import PRISMShieldQuery
from shield.shield_wrapper import map_to_abstract_action, abstract_action_to_continuous

logger = logging.getLogger(__name__)

BaseAgent = Union[PPOAgent, SACAgent]


class ShieldedAgent:
    """
    RL agent + probabilistic shield composite.

    Parameters
    ----------
    base_agent : PPOAgent | SACAgent
        Pre-trained RL agent (used for action proposals).
    shield_query : PRISMShieldQuery
        Initialised shield query engine.
    safety_threshold : float
        P(collision) above which the shield intervenes (default 0.05).
    verbose : bool
        Print intervention messages to stdout.
    """

    def __init__(
        self,
        base_agent: BaseAgent,
        shield_query: PRISMShieldQuery,
        safety_threshold: float = 0.05,
        verbose: bool = False,
    ):
        self.agent             = base_agent
        self.shield            = shield_query
        self.safety_threshold  = safety_threshold
        self.verbose           = verbose
        self._abstraction      = StateAbstraction()

        # Statistics
        self.total_queries:         int = 0
        self.total_interventions:   int = 0
        self.intervention_history:  List[Dict[str, Any]] = []

    # ── Main API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict a safe action for the given observation.

        1. Query the base RL agent for its preferred action.
        2. Abstract obs → (d, f, t).
        3. Check safety via PRISM shield.
        4. Override if unsafe.

        Parameters
        ----------
        obs : (OBS_DIM,) — current environment observation
        deterministic : bool — whether the RL agent uses its deterministic policy

        Returns
        -------
        action : (ACTION_DIM,) — final safe action
        """
        # Step 1: Get RL agent's proposed action
        proposed_action = self.agent.predict(obs, deterministic=deterministic)

        # Step 2: Abstract state
        abstract_state = self._abstraction.abstract(obs)

        # Step 3: Classify the proposed action
        abstract_proposed = map_to_abstract_action(proposed_action, obs)

        # Step 4: Safety check
        p_collision = self.shield.collision_prob(abstract_state, abstract_proposed)
        self.total_queries += 1

        if p_collision <= self.safety_threshold:
            return proposed_action  # safe → pass through

        # Step 5: Intervention — find least-restrictive safe action
        self.total_interventions += 1
        safe_set = self.shield.safe_action_set(abstract_state)

        if not safe_set:
            # Conservative fallback: hold position
            safe_abstract = 1   # ACTION_HOLD
            if self.verbose:
                logger.warning(
                    f"All actions unsafe at {abstract_state}. Holding position."
                )
        else:
            # Prefer approach if in safe set, else hold, else retreat
            preference = [abstract_proposed, 0, 1, 2]
            safe_abstract = next((a for a in preference if a in safe_set), safe_set[0])

        safe_action = abstract_action_to_continuous(safe_abstract, obs)

        if self.verbose:
            logger.info(
                f"Shield | state={abstract_state} | "
                f"P(col)={p_collision:.3f} > {self.safety_threshold} | "
                f"proposed={abstract_proposed} → safe={safe_abstract}"
            )

        self.intervention_history.append({
            "abstract_state":   abstract_state,
            "proposed_abstract": abstract_proposed,
            "safe_abstract":    safe_abstract,
            "p_collision":      p_collision,
        })

        return safe_action

    # ── Batch prediction ─────────────────────────────────────────────────────

    def predict_batch(
        self,
        obs_batch: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict safe actions for a batch of observations.

        Parameters
        ----------
        obs_batch : (n, OBS_DIM)

        Returns
        -------
        actions : (n, ACTION_DIM)
        """
        return np.stack([
            self.predict(obs, deterministic=deterministic)
            for obs in obs_batch
        ])

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def intervention_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_interventions / self.total_queries

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_queries":       self.total_queries,
            "total_interventions": self.total_interventions,
            "intervention_rate":   self.intervention_rate,
        }

    def reset_stats(self):
        self.total_queries       = 0
        self.total_interventions = 0
        self.intervention_history = []

    # ── Convenience ───────────────────────────────────────────────────────────

    def set_threshold(self, threshold: float):
        """Dynamically adjust the safety threshold (e.g., for ablation studies)."""
        self.safety_threshold = threshold
        self.shield.safety_threshold = threshold

    def disable_shield(self):
        """Temporarily disable the shield (set threshold = 1.0)."""
        self.set_threshold(1.0)

    def enable_shield(self, threshold: float = 0.05):
        """Re-enable the shield."""
        self.set_threshold(threshold)
