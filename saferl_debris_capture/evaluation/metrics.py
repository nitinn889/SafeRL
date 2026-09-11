"""
metrics.py
==========
Metric definitions and statistical analysis tools for the SafeRL benchmark.

Provides:
  - SafeRLMetrics: computes the core metrics used in the paper
  - Statistical significance testing (Mann-Whitney U, bootstrap CI)
  - Convergence curve analysis (learning curves)
  - Shield intervention analysis

All metrics are defined in line with the SafeRL literature:
  - Achiam et al. (2017). "Constrained Policy Optimisation." ICML.
  - Ray et al. (2019). "Benchmarking Safe Exploration in Deep Reinforcement
    Learning." arXiv.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats


# ─────────────────────────────────────────────────────────────────────────────
# Core metric definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeMetrics:
    """Per-episode metrics vector."""
    reward:         float
    cost:           float             # cumulative safety cost (collision count)
    success:        bool
    episode_length: int
    capture_step:   Optional[int]
    interventions:  int


@dataclass
class AggregatedMetrics:
    """Summary statistics over N episodes."""
    # Task performance
    success_rate:          float
    mean_reward:           float
    std_reward:            float
    median_reward:         float
    mean_episode_length:   float
    mean_capture_step:     float

    # Safety
    mean_cost:             float
    std_cost:              float
    collision_rate:        float       # fraction of episodes with cost > 0
    constraint_violation_rate: float  # fraction above cost threshold

    # Shield
    intervention_rate:     float
    mean_interventions_per_ep: float

    # Statistical
    reward_95ci:           Tuple[float, float]
    cost_95ci:             Tuple[float, float]


# ─────────────────────────────────────────────────────────────────────────────
# SafeRL Metrics class
# ─────────────────────────────────────────────────────────────────────────────

class SafeRLMetrics:
    """
    Computes and compares SafeRL performance metrics.

    Parameters
    ----------
    cost_threshold : float
        Acceptable cumulative cost per episode (for constraint violation rate).
    n_bootstrap : int
        Bootstrap samples for confidence intervals.
    ci_level : float
        Confidence interval level (0.95 = 95%).
    """

    def __init__(
        self,
        cost_threshold: float = 1.0,    # 1 collision per episode is the constraint
        n_bootstrap: int = 10000,
        ci_level: float = 0.95,
    ):
        self.cost_threshold = cost_threshold
        self.n_bootstrap    = n_bootstrap
        self.ci_level       = ci_level

    # ── Main API ─────────────────────────────────────────────────────────────

    def compute(self, episodes: List[EpisodeMetrics]) -> AggregatedMetrics:
        """
        Compute all metrics for a list of episodes.

        Parameters
        ----------
        episodes : List[EpisodeMetrics]

        Returns
        -------
        AggregatedMetrics
        """
        rewards       = np.array([e.reward       for e in episodes])
        costs         = np.array([e.cost         for e in episodes])
        lengths       = np.array([e.episode_length for e in episodes])
        successes     = np.array([e.success      for e in episodes])
        interventions = np.array([e.interventions for e in episodes])
        capture_steps = [e.capture_step for e in episodes if e.capture_step is not None]
        total_steps   = int(lengths.sum())

        return AggregatedMetrics(
            # Task performance
            success_rate          = float(successes.mean()),
            mean_reward           = float(rewards.mean()),
            std_reward            = float(rewards.std()),
            median_reward         = float(np.median(rewards)),
            mean_episode_length   = float(lengths.mean()),
            mean_capture_step     = float(np.mean(capture_steps)) if capture_steps else float("nan"),

            # Safety
            mean_cost             = float(costs.mean()),
            std_cost              = float(costs.std()),
            collision_rate        = float((costs > 0).mean()),
            constraint_violation_rate = float((costs > self.cost_threshold).mean()),

            # Shield
            intervention_rate     = float(interventions.sum() / total_steps) if total_steps > 0 else 0.0,
            mean_interventions_per_ep = float(interventions.mean()),

            # Statistical
            reward_95ci           = self._bootstrap_ci(rewards),
            cost_95ci             = self._bootstrap_ci(costs),
        )

    # ── Comparison / significance testing ────────────────────────────────────

    def compare(
        self,
        episodes_a: List[EpisodeMetrics],
        episodes_b: List[EpisodeMetrics],
        label_a: str = "A",
        label_b: str = "B",
    ) -> Dict:
        """
        Compare two conditions using Mann-Whitney U test.

        Returns
        -------
        dict with keys: reward_pvalue, cost_pvalue, success_pvalue,
                        reward_cohen_d, cost_improvement
        """
        rewards_a = np.array([e.reward for e in episodes_a])
        rewards_b = np.array([e.reward for e in episodes_b])
        costs_a   = np.array([e.cost   for e in episodes_a])
        costs_b   = np.array([e.cost   for e in episodes_b])
        succ_a    = np.array([e.success for e in episodes_a], dtype=float)
        succ_b    = np.array([e.success for e in episodes_b], dtype=float)

        r_stat, r_p = stats.mannwhitneyu(rewards_a, rewards_b, alternative="two-sided")
        c_stat, c_p = stats.mannwhitneyu(costs_a, costs_b, alternative="two-sided")

        return {
            f"{label_a}_vs_{label_b}": {
                "reward_pvalue":     float(r_p),
                "cost_pvalue":       float(c_p),
                "reward_cohen_d":    self._cohen_d(rewards_a, rewards_b),
                "cost_improvement":  float(costs_a.mean() - costs_b.mean()),
                "success_delta":     float(succ_b.mean() - succ_a.mean()),
                "significant_reward": r_p < 0.05,
                "significant_cost":   c_p < 0.05,
            }
        }

    # ── Learning curve analysis ───────────────────────────────────────────────

    @staticmethod
    def learning_curve_smoothed(
        values: List[float],
        window: int = 50,
    ) -> np.ndarray:
        """
        Apply a running-mean smoothing to a learning curve.

        Parameters
        ----------
        values : list of per-episode values
        window : smoothing window size

        Returns
        -------
        smoothed : np.ndarray
        """
        arr = np.array(values, dtype=np.float64)
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="valid")

    @staticmethod
    def cumulative_cost_curve(costs: List[float]) -> np.ndarray:
        """Return cumulative cost curve (for plotting)."""
        return np.cumsum(costs)

    @staticmethod
    def intervention_heatmap(
        interventions: List[dict],
    ) -> np.ndarray:
        """
        Build a (5 × 3 × 3) heatmap of intervention counts per abstract state.

        Parameters
        ----------
        interventions : list of dicts with key 'abstract_state'

        Returns
        -------
        heatmap : (5, 3, 3) int array indexed by [d, f, t]
        """
        heatmap = np.zeros((5, 3, 3), dtype=int)
        for iv in interventions:
            state = iv.get("abstract_state")
            if state is not None:
                heatmap[state.d, state.f, state.t] += 1
        return heatmap

    # ── Statistical helpers ───────────────────────────────────────────────────

    def _bootstrap_ci(self, arr: np.ndarray) -> Tuple[float, float]:
        """Bootstrap percentile confidence interval."""
        boot_means = [
            np.mean(np.random.choice(arr, size=len(arr), replace=True))
            for _ in range(self.n_bootstrap)
        ]
        alpha = (1 - self.ci_level) / 2
        return (
            float(np.percentile(boot_means, alpha * 100)),
            float(np.percentile(boot_means, (1 - alpha) * 100)),
        )

    @staticmethod
    def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
        """Cohen's d effect size between two samples."""
        pooled_std = np.sqrt(
            (a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2
        )
        if pooled_std == 0:
            return 0.0
        return float((a.mean() - b.mean()) / pooled_std)
