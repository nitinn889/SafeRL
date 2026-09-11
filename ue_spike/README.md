# UE Python-API viability spike (Goal A, Phase 2)

## Status: scaffolded, NOT yet executed

This spike is fully written but has not been run against a live editor in
this session. The only UE 5.8 install found on this machine lives on an
NTFS partition (`/dev/nvme0n1p3`) that is not mounted, and mounting it
requires `sudo`, which needs an interactive password this session cannot
supply. Everything below is ready to run as soon as the drive is mounted
(or `UE_EDITOR_CMD` is pointed at any other UE 5.8 `UnrealEditor-Cmd`).

To unblock, run once, in your own terminal:
```bash
sudo mkdir -p /mnt/bigdata
sudo mount -t ntfs-3g -o remove_hiberfile,rw,uid=1000,gid=1000 /dev/nvme0n1p3 /mnt/bigdata
```
Then:
```bash
./ue_spike/run_spike.sh
```
This writes `ue_spike/ue_spike_results.json` with the measured steps/sec.
Update the go/no-go numbers in the main [README.md](../README.md) phase 2
section with the real figures once you have them -- the numbers currently
there are explicitly marked as unmeasured placeholders.

## What's here

- `SafeRLUESpike.uproject` -- blank, content-only UE 5.8 project (no C++
  module, so no build step) with `PythonScriptPlugin` and
  `EditorScriptingUtilities` enabled.
- `Content/Python/init_unreal.py` -- auto-run on engine startup by UE's
  Python plugin convention; just confirms the bridge is alive.
- `Content/Python/ue_bridge.py` -- the actual `reset()`/`step()` loop
  against a placeholder actor (a `StaticMeshActor` using the engine's
  built-in sphere mesh), plus `run_benchmark()` which runs 500 steps and
  times them.
- `Content/Python/ue_gym_env.py` -- a `gymnasium.Env` wrapper
  (`UESafeNavSpikeEnv`) around the bridge, same interface shape as
  `saferl.env.base_env.SafeNav3DEnv` (`Discrete(4)` actions, `Box`
  observation). No shield, no PPO -- just the loop, per the phase 2 spike
  scope.
- `run_spike.sh` -- launches `UnrealEditor-Cmd` headless
  (`-nullrhi -unattended`) and runs the benchmark.

## Design decisions (and why)

**In-process (Editor Python) vs. an external process + Remote Execution
plugin:** chose in-process. The `unreal` module (actor spawn/transform
calls) only exists inside the engine process. Driving it from an external
process instead means going through UE's Remote Execution plugin
(multicast UDP), which adds a serialization + IPC hop per call and needs
the editor already running with that plugin turned on -- more moving
parts, noisier measurement, for a first pass at "is this viable at all."
In-process, headless, via `UnrealEditor-Cmd -run=pythonscript`, is UE's
own documented automation pattern.

**What the benchmark actually measures:** Python↔engine round-trip cost
for actor transform set/get plus one `editor_tick()` per step -- the
throughput floor any RL loop sits on top of. Position/velocity are
integrated kinematically in Python and pushed to the actor each step,
then read back through the engine API (a genuine round trip, not just a
local Python variable). It does **not** yet exercise UE's rigid-body
physics solver -- that only ticks in a running PIE/game world, not the
bare editor world this spike uses. If the go/no-go lands on "build in
UE," physics-tick throughput under PIE is the next measurement to take,
and is a bigger unknown than what's measured here.

## Arithmetic to fill in once you have a number

`saferl/configs/default.yaml` currently trains for `timesteps: 30000`
(matching the original script). At measured rate `R` steps/sec:
```
30000 / R = seconds for one 30k-step PPO run
```
PPO runs in this space typically want far more than 30k steps once the
shield/reward are non-trivial (tens of thousands to low millions). Multiply
accordingly before deciding UE is fast enough to train against directly.

See the main [README.md](../README.md) phase 2 section for the actual
go/no-go recommendation once the number above is filled in.
