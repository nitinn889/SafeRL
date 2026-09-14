"""Deliberate collision-course scenarios, run against each shield variant.

Phase 5's claim is that the least-restrictive shield is *safer* than the
phase-1 random-replacement placeholder, not merely tidier. Randomly spawned
debris rarely produce a genuine collision course, so this module constructs
them: pick an intercept time, propagate the agent's un-shielded trajectory
analytically, and place each hazard so that it arrives at that point at that
time. The encounter is then guaranteed, and the only variable is the shield.

Run:  python -m saferl.eval.stress_test
"""
import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pybullet as p

from saferl.env.base_env import (
    ACTION_THRUST_DIRS_3D, AGENT_MASS, OBS_HEADER_LEN, OBS_PER_HAZARD, SafeNav3DEnv,
    thrust_dirs,
)
from saferl.shield.safety_shield import (
    DEFAULT_ACCEL, RandomReplacementShield, SafetyShield,
)

AGENT_REST_Z = 0.25   # where the 0.5-scaled sphere settles on the plane
HAZARD_Z = 0.5        # env default spawn height for debris


@dataclass
class HazardSpec:
    """A hazard defined by *where along the path it meets the agent*.

    The encounter is specified as `intercept_distance` units travelled along
    the agent's un-shielded path, NOT as a time. Phase 6 changed force_mag
    from 12N to 48N and every time-specified encounter silently moved outside
    the 10-unit play area, where boundary reflection destroyed the designed
    trajectory and the scenario quietly stopped testing anything. Distance
    along the path is invariant to thrust; the harness solves for the time.

    The hazard arrives from direction `approach_from` at `speed` units/sec;
    speed=0 parks a stationary obstacle on the path.
    """
    intercept_distance: float
    approach_from: tuple      # side the hazard comes from, e.g. (0, -1, 0) = below
    speed: float
    offset: tuple = (0.0, 0.0, 0.0)
    # Added to the computed start position. Used by the `surrounded` scenario,
    # where the hazards are not intercepting anything -- they are parked in a
    # ring around the agent -- so the offset *is* the placement.


@dataclass
class Scenario:
    name: str
    description: str
    agent_pos: tuple
    agent_vel: tuple
    intent_action: int        # the constant action the "policy" wants
    hazards: list
    steps: int = 90


def _agent_position_at(scn, t, accel):
    """Un-shielded agent position at time t: p0 + v0 t + 0.5 a t^2."""
    p0 = np.array(scn.agent_pos, dtype=np.float64)
    v0 = np.array(scn.agent_vel, dtype=np.float64)
    a = ACTION_THRUST_DIRS_3D[scn.intent_action] * accel
    return p0 + v0 * t + 0.5 * a * t * t


def _time_to_travel(scn, distance, accel):
    """When the un-shielded agent has covered `distance` along its intent axis.

    Solves 0.5*a*t^2 + v*t - distance = 0 for the positive root, where v is
    the agent's initial speed along that axis. This is what makes a scenario
    survive a change in force_mag: the geometry is pinned in the play area and
    the timing falls out of the physics.
    """
    if distance <= 0:
        return 0.0
    u = ACTION_THRUST_DIRS_3D[scn.intent_action]
    v = float(np.dot(np.array(scn.agent_vel, dtype=np.float64), u))
    if accel <= 0:
        if v <= 0:
            raise ValueError(f"{scn.name}: agent never covers {distance} units")
        return distance / v
    return float((-v + np.sqrt(v * v + 2.0 * accel * distance)) / accel)


def build_scenarios(dims=2):
    """Five encounters, chosen to separate distance-only from velocity-aware.

    `closing_head_on` and `crossing_*` are invisible to a distance-only
    trigger until late, because the danger is in the relative velocity.
    `parked_obstacle` is the control: a distance-only shield handles it fine.
    `pincer` is built so no single action stays clear -- it exercises the
    boxed-in fallback.

    `dims=3` returns the free-flight set instead (see _free_flight_scenarios).
    Actions are looked up in ACTION_THRUST_DIRS_3D throughout this module: its
    first four rows are the planar table, so planar scenarios are unaffected
    and a substituted +Z/-Z in 3D still resolves.
    """
    if dims == 3:
        return _free_flight_scenarios()
    return [
        Scenario(
            name="closing_head_on",
            description="debris drifting straight back down the agent's +X path",
            agent_pos=(1.0, 5.0, AGENT_REST_Z), agent_vel=(1.0, 0.0, 0.0),
            intent_action=3,
            hazards=[HazardSpec(intercept_distance=5.0, approach_from=(1, 0, 0), speed=1.2)],
        ),
        Scenario(
            name="crossing_from_right",
            description="debris cutting across the path from +Y, perpendicular",
            agent_pos=(1.0, 5.0, AGENT_REST_Z), agent_vel=(1.0, 0.0, 0.0),
            intent_action=3,
            hazards=[HazardSpec(intercept_distance=5.0, approach_from=(0, 1, 0), speed=1.2)],
        ),
        Scenario(
            name="crossing_from_left",
            description="mirror of crossing_from_right; catches a shield that "
                        "always dodges the same way",
            agent_pos=(1.0, 5.0, AGENT_REST_Z), agent_vel=(1.0, 0.0, 0.0),
            intent_action=3,
            hazards=[HazardSpec(intercept_distance=5.0, approach_from=(0, -1, 0), speed=1.2)],
        ),
        Scenario(
            name="parked_obstacle",
            description="control case: a stationary hazard on the path, which "
                        "the distance-only trigger also sees",
            agent_pos=(1.0, 5.0, AGENT_REST_Z), agent_vel=(1.0, 0.0, 0.0),
            intent_action=3,
            hazards=[HazardSpec(intercept_distance=5.5, approach_from=(1, 0, 0), speed=0.0)],
        ),
        Scenario(
            name="surrounded",
            description="four stationary hazards ringing the start position just "
                        "inside safe_dist: no action is safe, so every step is a "
                        "fallback. The agent should still survive -- fallback is "
                        "'buy time', not 'give up'.",
            agent_pos=(5.0, 5.0, AGENT_REST_Z), agent_vel=(0.0, 0.0, 0.0),
            intent_action=3,
            hazards=[HazardSpec(intercept_distance=0.0, approach_from=d, speed=0.0,
                                offset=tuple(2.0 * np.array(d, dtype=float)))
                     for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))],
            steps=60,
        ),
        Scenario(
            name="pincer",
            description="two hazards converging from both flanks plus one ahead; "
                        "no action stays clear, so the fallback must fire",
            agent_pos=(1.0, 5.0, AGENT_REST_Z), agent_vel=(1.0, 0.0, 0.0),
            intent_action=3,
            hazards=[
                HazardSpec(intercept_distance=5.0, approach_from=(0, 1, 0), speed=1.4),
                HazardSpec(intercept_distance=5.0, approach_from=(0, -1, 0), speed=1.4),
                HazardSpec(intercept_distance=6.5, approach_from=(1, 0, 0), speed=1.0),
            ],
        ),
    ]


FREE_FLIGHT_Z = 5.0   # mid-volume: free flight has no floor to rest on


def _free_flight_scenarios():
    """3D encounters: the planar set's construction plus the vertical axis.

    Hazards are timed to meet the agent's un-shielded +X path exactly as in
    the planar set, with the agent mid-volume. The vertical cases are the
    point: a hazard arriving from above or below is outside anything a planar
    shield models, `vertical_pincer` is only escapable sideways, and
    `surrounded_6` boxes the agent in on every axis.
    """
    start = (1.0, 5.0, FREE_FLIGHT_Z)
    cruise = (1.0, 0.0, 0.0)
    plus_x = 3

    def crossing(name, description, approach_from):
        return Scenario(
            name=name, description=description,
            agent_pos=start, agent_vel=cruise, intent_action=plus_x,
            hazards=[HazardSpec(intercept_distance=5.0, approach_from=approach_from,
                                speed=1.2)],
        )

    axes6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    return [
        crossing("closing_head_on_3d",
                 "debris drifting straight back down the +X path, mid-volume",
                 (1, 0, 0)),
        crossing("crossing_from_above", "debris dropping across the path from +Z",
                 (0, 0, 1)),
        crossing("crossing_from_below", "debris rising across the path from -Z",
                 (0, 0, -1)),
        crossing("oblique_crossing",
                 "debris cutting across from above and to the side at 45 degrees",
                 (0, 1, 1)),
        Scenario(
            name="vertical_pincer",
            description="hazards converging from above and below plus one "
                        "ahead; the only escape is sideways",
            agent_pos=start, agent_vel=cruise, intent_action=plus_x,
            hazards=[
                HazardSpec(intercept_distance=5.0, approach_from=(0, 0, 1), speed=1.4),
                HazardSpec(intercept_distance=5.0, approach_from=(0, 0, -1), speed=1.4),
                HazardSpec(intercept_distance=6.5, approach_from=(1, 0, 0), speed=1.0),
            ],
        ),
        Scenario(
            name="surrounded_6",
            description="six stationary hazards boxing the agent in on every "
                        "axis just inside safe_dist: every step is a fallback",
            agent_pos=(5.0, 5.0, FREE_FLIGHT_Z), agent_vel=(0.0, 0.0, 0.0),
            intent_action=plus_x,
            hazards=[HazardSpec(intercept_distance=0.0, approach_from=d, speed=0.0,
                                offset=tuple(2.0 * np.array(d, dtype=float)))
                     for d in axes6],
            steps=60,
        ),
    ]


def _place(env, scn, accel):
    """Teleport agent and hazards into the scenario's opening state."""
    cid = env._client
    p.resetBasePositionAndOrientation(env.agent_id, list(scn.agent_pos),
                                      [0, 0, 0, 1], physicsClientId=cid)
    p.resetBaseVelocity(env.agent_id, linearVelocity=list(scn.agent_vel),
                        angularVelocity=[0, 0, 0], physicsClientId=cid)

    for i, spec in enumerate(scn.hazards):
        t_meet = _time_to_travel(scn, spec.intercept_distance, accel)
        meet = _agent_position_at(scn, t_meet, accel)
        u = np.array(spec.approach_from, dtype=np.float64)
        u = u / (np.linalg.norm(u) or 1.0)
        vel = -u * spec.speed                       # travels toward the meeting point
        start = meet - vel * t_meet + np.array(spec.offset, dtype=np.float64)
        _assert_in_play_area(scn, env.size, meet, start, env._axes)
        if env.dims == 3:
            h_pos = [float(c) for c in start]
            env.hazard_velocities[i] = [float(c) for c in vel]
        else:
            h_pos = [float(start[0]), float(start[1]), HAZARD_Z]
            env.hazard_velocities[i] = [float(vel[0]), float(vel[1]), 0.0]
        env.hazard_positions[i] = h_pos
        p.resetBasePositionAndOrientation(env._hazard_ids[i], h_pos,
                                          [0, 0, 0, 1], physicsClientId=cid)


def _assert_in_play_area(scn, size, meet, start, axes=(0, 1)):
    """Fail loudly if a scenario places its encounter outside the field.

    Debris reflect off the [0, size] boundary, so a hazard placed outside it
    is immediately bounced off its designed course and the scenario silently
    degrades into noise. Phase 6 hit exactly that, so it is now an error
    rather than a quietly passing test.
    """
    for label, pt in (("intercept point", meet), ("hazard start", start)):
        if not all(0.0 <= float(pt[ax]) <= size for ax in axes):
            raise ValueError(
                f"scenario {scn.name!r}: {label} {np.round(pt[:len(axes)], 2).tolist()} "
                f"is outside the 0..{size} play area -- retune "
                f"intercept_distance for the current force_mag"
            )


def _toward_hazard(obs, executed_action, hazard_index):
    """Shield-agnostic geometric audit of one intervention.

    True when the substituted thrust has a positive component along the
    direction from the agent to the hazard that triggered -- i.e. the shield
    is accelerating the agent *at* the thing it is supposed to avoid. No
    lookahead model is involved, so it judges both shields on equal terms.
    """
    if hazard_index is None or hazard_index < 0:
        return False
    base = OBS_HEADER_LEN + hazard_index * OBS_PER_HAZARD
    if base + 3 > len(obs):
        return False
    to_hazard = np.asarray(obs[base:base + 3], dtype=np.float64) - np.asarray(obs[0:3], dtype=np.float64)
    n = np.linalg.norm(to_hazard)
    if n < 1e-9:
        return True
    return bool(np.dot(ACTION_THRUST_DIRS_3D[int(executed_action)], to_hazard / n) > 1e-9)


def run_episode(scn, shield, auditor, cfg_env, seed=0):
    """One scenario rollout under one shield (or None for the un-shielded arm)."""
    np.random.seed(seed)
    env_kwargs = dict(cfg_env)
    env_kwargs.update(max_hazards=len(scn.hazards), curriculum=False,
                      max_episode_steps=scn.steps)
    env = SafeNav3DEnv(render_mode="direct", **env_kwargs)
    accel = env.force_mag / AGENT_MASS
    env.reset()
    _place(env, scn, accel)
    obs = env._get_obs()

    rec = dict(scenario=scn.name, collided=False, steps=0, interventions=0,
               fallback_interventions=0, toward_hazard_interventions=0,
               self_endangering_interventions=0, intervention_caused_collisions=0,
               min_distance=float("inf"))
    # steps at which a self-endangering / toward-hazard intervention executed
    flagged_steps = []

    for t in range(scn.steps):
        action = scn.intent_action
        if shield is not None:
            action, intervened = shield.check_and_fix(obs, action)
            if intervened:
                rec["interventions"] += 1
                d = shield.last_decision
                if d.kind == "fallback":
                    rec["fallback_interventions"] += 1
                toward = _toward_hazard(obs, action, d.hazard_index)
                if toward:
                    rec["toward_hazard_interventions"] += 1
                # auditor: was the substituted action itself predicted to
                # breach, when a safe alternative was on the table?
                safe_set = auditor.safe_actions(obs)
                if safe_set and action not in safe_set:
                    rec["self_endangering_interventions"] += 1
                    flagged_steps.append(t)
                elif toward:
                    flagged_steps.append(t)

        obs, reward, done, truncated, info = env.step(int(action))
        rec["steps"] = t + 1
        for h in env.hazard_positions:
            rec["min_distance"] = min(rec["min_distance"],
                                      float(np.linalg.norm(obs[0:3] - np.array(h))))
        if info.get("cost", 0) == 1:
            rec["collided"] = True
            horizon = getattr(auditor, "lookahead_steps", 20)
            if any(t - s <= horizon for s in flagged_steps):
                rec["intervention_caused_collisions"] = 1
            break
        if done or truncated:
            break

    env.close()
    return rec


def run_comparison(trials=25, cfg=None, lookahead_steps=None, safe_dist=None):
    """Run every scenario under: no shield, old random shield, new shield.

    lookahead_steps / safe_dist default to the config rather than to literals,
    so the harness follows a physics change instead of quietly measuring the
    previous tune.
    """
    from saferl.config import load_config
    cfg = cfg or load_config()
    if lookahead_steps is None:
        lookahead_steps = cfg["shield"]["lookahead_steps"]
    if safe_dist is None:
        safe_dist = cfg["shield"]["safe_dist"]
    cfg_env = dict(cfg["env"])
    accel = cfg_env["force_mag"] / AGENT_MASS
    step_dt = cfg_env["sim_substeps"] / 240.0
    dims = cfg_env.get("dims", 2)
    dirs = thrust_dirs(dims)

    def new_shield():
        return SafetyShield(safe_dist=safe_dist, lookahead_steps=lookahead_steps,
                            accel=accel, step_dt=step_dt, thrust_dirs=dirs)

    arms = {
        "no_shield": lambda seed: None,
        "old_random": lambda seed: RandomReplacementShield(
            safe_dist=safe_dist, rng=np.random.RandomState(seed),
            n_actions=len(dirs)),
        "new_least_restrictive": lambda seed: new_shield(),
    }

    results = {}
    for scn in build_scenarios(dims):
        results[scn.name] = {}
        for arm, factory in arms.items():
            runs = [run_episode(scn, factory(seed), new_shield(), cfg_env, seed=seed)
                    for seed in range(trials)]
            results[scn.name][arm] = _aggregate(runs, trials)
    return results


def _aggregate(runs, trials):
    keys = ["interventions", "fallback_interventions", "toward_hazard_interventions",
            "self_endangering_interventions"]
    return {
        "trials": trials,
        "collisions": sum(r["collided"] for r in runs),
        "intervention_caused_collisions": sum(r["intervention_caused_collisions"] for r in runs),
        "mean_steps_survived": round(float(np.mean([r["steps"] for r in runs])), 1),
        "mean_min_distance": round(float(np.mean([r["min_distance"] for r in runs])), 3),
        **{f"mean_{k}": round(float(np.mean([r[k] for r in runs])), 2) for k in keys},
    }


def format_table(results):
    hdr = (f"{'scenario':<22} {'arm':<22} {'coll':>5} {'ivc':>4} {'ivs':>6} "
           f"{'fb':>5} {'toward':>7} {'unsafe':>7} {'minD':>6}")
    lines = [hdr, "-" * len(hdr)]
    for scn, arms in results.items():
        for arm, a in arms.items():
            lines.append(
                f"{scn:<22} {arm:<22} {a['collisions']:>3}/{a['trials']:<2} "
                f"{a['intervention_caused_collisions']:>4} "
                f"{a['mean_interventions']:>6.1f} {a['mean_fallback_interventions']:>5.1f} "
                f"{a['mean_toward_hazard_interventions']:>7.1f} "
                f"{a['mean_self_endangering_interventions']:>7.1f} "
                f"{a['mean_min_distance']:>6.2f}")
        lines.append("")
    lines.append("coll=collisions/trials  ivc=collisions implicated by an intervention")
    lines.append("ivs=mean interventions  fb=mean fallback (boxed-in) interventions")
    lines.append("toward=mean interventions that thrust toward the triggering hazard")
    lines.append("unsafe=mean interventions the auditor rates unsafe when a safe action existed")
    lines.append("minD=mean closest approach to any hazard over the episode")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Old vs new shield on collision-course scenarios.")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--lookahead-steps", type=int, default=None,
                help="default: config shield.lookahead_steps")
    ap.add_argument("--safe-dist", type=float, default=None,
                help="default: config shield.safe_dist")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--config", default=None,
                    help="Config YAML (default: planar default.yaml; "
                         "saferl/configs/space3d.yaml runs the 3D scenario set)")
    args = ap.parse_args()

    from saferl.config import load_config
    results = run_comparison(trials=args.trials, cfg=load_config(args.config),
                             lookahead_steps=args.lookahead_steps,
                             safe_dist=args.safe_dist)
    print(format_table(results))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
