# UE Python-API viability spike (Goal A, Phase 2)

## Status: executed. Result: inconclusive on the number that actually matters -- see recommendation below.

The drive holding the UE 5.8 install (`/dev/nvme0n1p3`, mounted at
`/mnt/bigdata`) is now mounted, and the spike ran against the real engine.
Two things came out of it, and they point in different directions -- read
both before treating either as "the number."

## What ran, and what it measured

### 1. Commandlet run (`./run_spike.sh`) -- ~180,000 "steps"/sec

```bash
UE_EDITOR_CMD=/mnt/bigdata/unreal_engine/Engine/Binaries/Linux/UnrealEditor-Cmd ./ue_spike/run_spike.sh
```
`UnrealEditor-Cmd <project> -run=pythonscript -script=ue_bridge.py -nullrhi -unattended`
loaded the project, ran `ue_bridge.py`'s `run_benchmark()` (500 steps
against a spawned `StaticMeshActor` -- position set, kinematic velocity
integrated in Python, position read back through the engine API each
step), and exited cleanly. Two runs: 500 steps in 0.00279s and 0.00277s,
i.e. **~179,000-181,000 "steps"/sec**, written to
[`ue_spike_results.json`](ue_spike_results.json). No crash, fully stable.

**This number is real but not the one you want for a go/no-go.** It is a
single Python script executing top to bottom inside a commandlet with
*no running frame loop at all* -- no tick, no physics, no rendering
between calls. It measures the floor cost of a Python call into UE's C++
actor-transform accessors (basically two property sets/gets), which
was never going to be the bottleneck. It says nothing about how fast UE
can actually advance a simulated frame, which is the number that
determines real training throughput.

### 2. Tick-driven run (`Content/Python/bench_tick_rate.py`) -- blocked

To get a number bound by an actual running engine loop instead of a
single-shot script, I registered `unreal.register_slate_post_tick_callback`
and tried to run one env step per real engine tick, launched as a
persistent (non-commandlet) process:
```bash
UnrealEditor-Cmd Project.uproject -ExecutePythonScript=bench_tick_rate.py -nullrhi -nosplash -nopause -log
```
Tried this four ways: `UnrealEditor-Cmd` and full `UnrealEditor`, each
with and without `-unattended`. **All four self-terminated within 1-2
engine ticks** (`Cmd: QUIT_EDITOR` / `UUnrealEdEngine::CloseEditor()`,
~10-20ms after the callback armed) -- not from my code (instrumented the
callback: it logged `tick count=1` and never reached the `count >= 500`
branch, no exception raised) but from the engine itself. Best read: a
bare `-nullrhi` editor process with no PIE/game session and no viewport
has nothing it considers worth staying alive for, and winds itself down
once the startup Python script returns control, regardless of a
registered per-tick callback.

**This is itself a real finding, not just a dead end:** driving training
steps from a persistently-running headless editor process isn't as simple
as "register a callback and go." Getting a genuine per-frame-tick number
(and a genuine training loop, later) needs either a running PIE session
or `-game` mode with an actual level/GameMode keeping the world ticking --
meaningfully more infrastructure than this spike's scope ("confirm the
loop is stable, don't build the full thing").

## Go/no-go recommendation

**I can't respons­ibly call this "fast enough" or "too slow" yet, because
the one number that would decide it -- real physics/render-tick throughput
under a running world -- is exactly the number the spike could not get to.**
What I can say:

- The `saferl/configs/default.yaml` default of `timesteps: 30000` would
  take **~0.17s** at the (not representative) 180k/s call-overhead number,
  or anywhere from a few seconds to tens of minutes at a genuine
  physics-tick rate, depending on whether headless UE ticks are
  throttled anywhere near real-time (30-120 fps) or run much faster
  unthrottled with nothing to render. That's a wide enough range that
  guessing isn't useful -- and PPO here will likely want well beyond 30k
  steps once the shield/reward move past placeholders, which only
  widens the gap between the good and bad ends of that range.
- The existing PyBullet path is a known, already-measured quantity: this
  session's regression check trained the full 30k-timestep run in ~28s at
  ~1090 fps, headless, no extra infrastructure required.
- Getting a UE-side apples-to-apples number needs a PIE/game-mode spike,
  which is real additional work, not a tweak to what's here.

**My recommendation:** default to the fallback -- train in the existing
PyBullet env (dynamics already match what a UE physics body would need to
approximate) and reserve UE for rendering/demo playback of a trained
policy, rather than as the training loop itself. That requires no new
engineering the codebase doesn't already have (`saferl/demo/run_demo.py`
already renders via PyBullet's GUI mode; swapping the renderer for UE
later is a smaller lift than making UE the live training loop now). If
direct-UE training is still wanted, the next concrete step is a follow-up
spike measuring step rate from inside a running PIE session (`-game
-nullrhi` with a minimal level/GameMode, driving steps from `BeginPlay`/
`Tick` rather than a registered editor callback) before committing further
design around it.

**This is my recommendation, not a decision made on your behalf** -- say
the word if you want the PIE-based follow-up spike instead of taking the
fallback.

## What's here

- `SafeRLUESpike.uproject` -- blank, content-only UE 5.8 project (no C++
  module, no build step) with `PythonScriptPlugin` and
  `EditorScriptingUtilities` enabled.
- `Content/Python/init_unreal.py` -- auto-run on engine startup by UE's
  Python plugin convention; confirms the bridge is alive.
- `Content/Python/ue_bridge.py` -- the `reset()`/`step()` loop against a
  placeholder actor, plus `run_benchmark()` (the commandlet-mode
  measurement above).
- `Content/Python/bench_tick_rate.py` -- the tick-driven measurement
  attempt (blocked, see above).
- `Content/Python/ue_gym_env.py` -- a `gymnasium.Env` wrapper
  (`UESafeNavSpikeEnv`) around the bridge, same interface shape as
  `saferl.env.base_env.SafeNav3DEnv`. Untested against a live loop beyond
  what's described above -- don't treat it as validated.
- `run_spike.sh` -- launches the commandlet-mode benchmark.
- `ue_spike_results.json` -- the actual measured commandlet-mode result.

Engine-generated `Binaries/`, `Intermediate/`, `Saved/`, `Config/` are
gitignored (see `ue_spike/.gitignore`) -- pure build/cache output, nothing
authored.
