"""Minimal reset()/step() loop driving a placeholder actor via UE's
in-process Editor Python API.

Why in-process (Editor Python) instead of an external process talking to
UE over the Remote Control / remote-execution plugin:
  - The `unreal` module (EditorLevelLibrary, actor spawn/transform calls)
    is only fully available inside the engine process itself.
  - The alternative -- an external Python process driving UE over the
    Remote Execution multicast-UDP plugin -- adds a serialization + IPC
    hop per call and requires the editor already running with that
    plugin enabled. That is more moving parts and a noisier measurement
    for a first viability check.
  - In-process, launched headless via UnrealEditor-Cmd, is the standard,
    well-documented pattern for UE automation scripts (see
    -ExecutePythonScript on the commandline) and is what we benchmark here.

What this does and doesn't measure:
  - It measures the Python <-> engine call round-trip for actor transform
    set/get -- i.e. the throughput floor any RL step loop would sit on top
    of, regardless of what drives the dynamics. Run via
    `-run=pythonscript` (a commandlet), the script executes once and
    exits; there is no running editor frame loop to advance between
    steps, so no per-step tick call is made here.
  - It does NOT yet run real rigid-body physics ticking (that requires a
    running PIE/game world, not just the editor world, and a frame loop
    actually advancing between Python calls). Position/velocity here are
    integrated kinematically in Python and pushed to the actor each step,
    then read back through the engine API -- a genuine Python<->engine
    round trip, just not UE's physics solver and not a real per-frame
    tick. If the go/no-go lands on "build in UE," PIE/game-world
    tick-driven throughput is the next, likely slower, thing to measure.
"""
import json
import time

import unreal

ACTOR_LABEL = "SafeRLSpikeAgent"
START_LOCATION = unreal.Vector(0.0, 0.0, 100.0)
DT = 1.0 / 30.0          # assumed step interval, seconds
FORCE_ACCEL = 400.0      # cm/s^2, stand-in for the 4 discrete thrust actions

# action -> (ax, ay) acceleration, mirrors SafeNav3DEnv's 4-action scheme
ACTION_TABLE = {
    0: (0.0,  FORCE_ACCEL),   # +Y
    1: (0.0, -FORCE_ACCEL),   # -Y
    2: (-FORCE_ACCEL, 0.0),   # -X
    3: (FORCE_ACCEL, 0.0),    # +X
}


def _actor_subsystem():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def _find_or_spawn_agent():
    subsys = _actor_subsystem()
    for actor in subsys.get_all_level_actors():
        if actor.get_actor_label() == ACTOR_LABEL:
            return actor

    sphere_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Sphere.Sphere")
    actor = subsys.spawn_actor_from_class(
        unreal.StaticMeshActor, START_LOCATION
    )
    actor.set_actor_label(ACTOR_LABEL)
    mesh_comp = actor.static_mesh_component
    mesh_comp.set_static_mesh(sphere_mesh)
    mesh_comp.set_mobility(unreal.ComponentMobility.MOVABLE)
    return actor


class UEBridge:
    """Kinematic reset()/step() bridge around one placeholder actor."""

    def __init__(self):
        self.actor = _find_or_spawn_agent()
        self.velocity = [0.0, 0.0, 0.0]

    def reset(self):
        self.actor.set_actor_location(START_LOCATION, False, False)
        self.velocity = [0.0, 0.0, 0.0]
        return self._read_obs()

    def step(self, action: int):
        ax, ay = ACTION_TABLE.get(action, (0.0, 0.0))
        self.velocity[0] += ax * DT
        self.velocity[1] += ay * DT

        loc = self.actor.get_actor_location()
        new_loc = unreal.Vector(
            loc.x + self.velocity[0] * DT,
            loc.y + self.velocity[1] * DT,
            loc.z,
        )
        self.actor.set_actor_location(new_loc, False, False)

        # Note: no explicit per-frame tick call here. Run via
        # `-run=pythonscript` (a commandlet), the script executes once and
        # exits -- there is no running editor frame loop to advance, so
        # this measures back-to-back actor transform set/get round trips
        # only. A running PIE/game world (see README) would tick on its
        # own between calls instead.
        return self._read_obs()

    def _read_obs(self):
        loc = self.actor.get_actor_location()  # round-trip read, not our local cache
        return {
            "position": [loc.x, loc.y, loc.z],
            "velocity": list(self.velocity),
        }


def run_benchmark(num_steps=500, out_path=None):
    bridge = UEBridge()
    bridge.reset()

    t0 = time.perf_counter()
    for i in range(num_steps):
        bridge.step(i % 4)
    elapsed = time.perf_counter() - t0

    result = {
        "num_steps": num_steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": num_steps / elapsed if elapsed > 0 else float("inf"),
    }
    unreal.log(f"[saferl-spike] {result}")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        unreal.log(f"[saferl-spike] wrote results to {out_path}")

    return result


if __name__ == "__main__":
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ue_spike_results.json")
    run_benchmark(num_steps=500, out_path=os.path.normpath(out_path))
