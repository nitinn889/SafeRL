"""Turn the phase 6 per-episode CSVs into the trend tables the README quotes.

Run:  python -m saferl.eval.phase6_report [--dir saferl/eval/phase6]

Deliberately reports deciles across training rather than a final number: the
claim phase 1 made was about a *trend* in interventions, and a single endpoint
cannot support or refute it.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

ARMS = ("unconstrained", "constrained")


def load(directory, arm):
    path = Path(directory) / f"{arm}_episodes.csv"
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            r[k] = float(v) if "." in v or "e" in v.lower() else int(v)
    return rows


def deciles(rows, n=10):
    out = []
    for i in range(n):
        chunk = rows[i * len(rows) // n:(i + 1) * len(rows) // n]
        if not chunk:
            continue
        out.append(dict(
            timestep=chunk[-1]["timestep"],
            task_reward=np.mean([r["task_reward"] for r in chunk]),
            rate=np.mean([r["intervention_rate"] for r in chunk]),
            ivs=np.mean([r["interventions"] for r in chunk]),
            steps=np.mean([r["steps"] for r in chunk]),
            goals=100 * np.mean([r["reached_goal"] for r in chunk]),
            coll=100 * np.mean([r["collided"] for r in chunk]),
            lam=np.mean([r["lam"] for r in chunk]),
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="saferl/eval/phase6")
    args = ap.parse_args()

    data = {arm: load(args.dir, arm) for arm in ARMS}
    for arm, rows in data.items():
        if not rows:
            print(f"[{arm}] no data")
            continue
        print(f"\n=== {arm} ({len(rows)} episodes) ===")
        print(f"{'timestep':>9} {'task_rew':>9} {'iv_rate':>8} {'ivs/ep':>8} "
              f"{'steps':>7} {'goal%':>6} {'coll%':>6} {'lambda':>7}")
        for d in deciles(rows):
            print(f"{d['timestep']:>9} {d['task_reward']:>9.1f} {d['rate']:>8.4f} "
                  f"{d['ivs']:>8.1f} {d['steps']:>7.1f} {d['goals']:>6.1f} "
                  f"{d['coll']:>6.1f} {d['lam']:>7.3f}")

    if all(data[a] for a in ARMS):
        print("\n=== final decile, constrained vs unconstrained ===")
        print(f"{'metric':<22} {'unconstrained':>14} {'constrained':>13} {'delta':>10}")
        u, c = deciles(data["unconstrained"])[-1], deciles(data["constrained"])[-1]
        for key, label, pct in (("rate", "interventions/step", True),
                                ("ivs", "interventions/episode", True),
                                ("task_reward", "task reward", False),
                                ("goals", "goal rate %", False),
                                ("coll", "collision rate %", False),
                                ("steps", "episode length", False)):
            delta = (f"{100*(c[key]-u[key])/u[key]:+.1f}%" if pct and u[key]
                     else f"{c[key]-u[key]:+.2f}")
            print(f"{label:<22} {u[key]:>14.4f} {c[key]:>13.4f} {delta:>10}")


if __name__ == "__main__":
    main()
