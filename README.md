This repository contains a Safe Reinforcement Learning (Safe RL) implementation using Probability Shields in a 3D navigation environment. The agent is trained using Proximal Policy Optimization (PPO) to navigate safely while avoiding hazards.

📖 Project Overview

In many RL environments, an agent may encounter unsafe states that can lead to failures or hazards. Probability Shields act as a safety layer, modifying or overriding unsafe actions in real-time to prevent collisions or unsafe behavior.

This project demonstrates:

A 3D Safe RL environment using PyBullet and Gymnasium.
Implementation of a SafetyShield wrapper to enforce safety constraints.
Training an RL agent using Stable Baselines3 PPO.
Collection of metrics including rewards, interventions, and safety violations.
Visualization of agent performance and safety compliance.

🛠 Features

Custom 3D navigation environment (SafeNav3DEnv)
Hazard generation with curriculum learning
Safety interventions via Probability Shields (SafetyShield)
Metrics collection (MetricsCallback) for rewards, costs, and interventions
Training with Stable Baselines3 PPO
Visualization of rewards, crashes, and interventions
3D demo of the trained agent

💻 Installation

Clone the repository and install dependencies:

git clone https://github.com/nitinn889/SafeRL.git



Dependencies:

Python >= 3.8
gymnasium
pybullet
numpy
torch
stable-baselines3
matplotlib
seaborn

⚙️ Usage

As of Phase 2 the code lives in the `saferl/` package (see the Phase 2
section below) instead of a single script. Install dependencies, then:

1. Train the agent
```
pip install -r requirements.txt
python -m saferl.training.train
```
This trains the PPO agent in the SafeNav3DEnv with safety interventions,
using `saferl/configs/default.yaml` for all hyperparameters. Metrics such
as rewards, cumulative crashes, and interventions are collected, and the
trained model is saved to `saferl_model.zip`.

2. Visualize training results
Graphs for rewards, total crashes, and interventions are saved to
`saferl_metrics.png` automatically at the end of training.

3. Run 3D demonstration
```
python -m saferl.demo.run_demo
```
Launches a PyBullet GUI showing the trained agent (loaded from
`saferl_model.zip`) navigating the environment while avoiding hazards.

4. Run tests
```
pytest tests/
```

📊 Results

After training, you can visualize:

Episode Rewards: Agent performance per episode.
Cumulative Crashes: Number of unsafe collisions over time.
Interventions: Number of times the Probability Shield intervened to prevent unsafe actions

🔬 References
Stable Baselines3 Documentation
PyBullet Documentation
Safe Reinforcement Learning with Probability Shields: Relevant research papers

⚡ Notes
Training time may vary depending on GPU/CPU.
Adjust total_timesteps in the training script to control training duration.
Safety interventions ensure the agent avoids hazards but may limit exploration initially.

---

## Phase 2 (2026-09-11): UE viability spike + repo restructure

### Goal A: UE Python-API viability spike -- executed, result inconclusive, my recommendation is the fallback

The drive holding UE 5.8 (`/dev/nvme0n1p3`, `/mnt/bigdata`) got mounted and
the spike ran for real. Full writeup with both measurements and the
reasoning: [`ue_spike/README.md`](ue_spike/README.md). Short version:

- **Commandlet-mode measurement (real, but not the number that matters):**
  `./ue_spike/run_spike.sh` ran 500 `reset()`/`step()` calls against a
  placeholder actor through UE's in-process Editor Python API (chosen
  over an external-process + Remote Execution setup -- fewer moving parts)
  headless via `UnrealEditor-Cmd -run=pythonscript -nullrhi -unattended`.
  Stable across two runs, no crash: **~179,000-181,000 "steps"/sec**. But
  this is a single Python script executing with no running frame loop at
  all between calls -- it's the floor cost of a Python-to-C++ property
  accessor call, not a per-simulated-frame rate. It was never going to be
  the bottleneck, and isn't informative for training-speed purposes on
  its own.
- **Tick-driven measurement (blocked):** tried to get a number bound by a
  real running engine loop by registering a per-tick callback and driving
  one step per actual engine tick, launched as a persistent (non-
  commandlet) process. Tried 4 ways (`UnrealEditor-Cmd`/`UnrealEditor` x
  with/without `-unattended`); all four self-terminated within 1-2 ticks
  regardless of the registered callback -- a bare `-nullrhi` editor with
  no PIE/game session apparently has nothing it considers worth staying
  alive for. That's a real finding: sustaining a persistent headless
  training loop needs a running PIE/`-game` session, not just a Python
  callback -- more infrastructure than this spike's scope.
- **Arithmetic:** `saferl/configs/default.yaml` trains for
  `timesteps: 30000`. At the (not representative) 180k/s call-overhead
  number that's ~0.17s; at a genuine physics-tick rate it could be
  anywhere from a few seconds to tens of minutes depending on whether
  headless UE throttles ticks toward real-time (30-120fps) or runs much
  faster unthrottled with nothing to render -- a spread wide enough that
  guessing isn't useful, and PPO here will likely want well beyond 30k
  steps once the shield/reward move past placeholders, widening the gap
  further.
- **For comparison, a known quantity:** this session's PyBullet regression
  check (below) trained the full 30k-timestep run in ~28s at ~1090 fps,
  headless, no extra infrastructure.

**My recommendation** (yours to accept or override, not decided
unilaterally): default to the fallback -- train in the existing PyBullet
env, use UE only for rendering/demo playback of a trained policy. That
needs no new engineering (`saferl/demo/run_demo.py` already renders via a
GUI env; swapping renderers later is a smaller lift than making UE the
live training loop now). If you'd rather pursue direct-UE training, the
concrete next step is a follow-up spike measuring step rate inside an
actual running PIE/`-game` session -- say so and that's the next thing to
build, not this fallback.

### Goal B: Repo restructure -- done

`2.py` (used as source of truth over the near-duplicate `Saferl.py`, which
is now removed) has been split into a package:
```
saferl/
  env/base_env.py        SafeNav3DEnv (PyBullet)
  shield/safety_shield.py SafetyShield, ShieldedEnv
  training/train.py       training entrypoint (python -m saferl.training.train)
  training/metrics.py     MetricsCallback
  configs/default.yaml    all hyperparameters (size, max_hazards, force_mag,
                           SAFE_DIST, TIMESTEPS, etc. -- no more magic numbers in code)
  demo/run_demo.py        visual demo entrypoint, loads a saved model
tests/test_env.py         reset/step shape checks + shield intervention tests
```
`2.py`'s three bugfixes are preserved as-is: obs tracked in `ShieldedEnv`
(not re-fetched from the base env), `make_env` is a factory function (not
a lambda), and the cost accumulator resets on episode boundaries in
`MetricsCallback`.

Two behavioral changes from splitting one script into a package: `train.py`
now saves the trained model to `saferl_model.zip` (so `run_demo.py` can
load it independently -- the original script kept `model` in memory and
ran the demo in the same process), and `SafetyShield.safe_dist` /
`force_mag` moved from hardcoded class attributes to constructor params
read from `configs/default.yaml`. No env physics, reward shaping, or shield
logic changed.

**Regression check, run this session:** `python -m saferl.training.train`
with the default config (30,000 timesteps, PyBullet DIRECT mode) completed
in ~28s at ~1090 fps, trained without errors, and saved a model. Both the
restructured version and the original untouched `2.py` produce **0
completed episodes** in that run and therefore no metrics plot -- verified
identical on both, so this is not a regression, but a pre-existing
characteristic worth flagging: the env has no episode step cap
(no `TimeLimit`), and with `SAFE_DIST=2.2` vs. `hazard_threshold=1.3` the
shield tends to steer the agent away before a hazard collision can ever
set `done=True`, while reaching the goal under an early-training policy is
rare enough that a single episode can run past 30k steps without
terminating either way. Worth deciding in Phase 3: add a max-episode-step
limit, or otherwise loosen why episodes rarely end.

`tests/test_env.py` (5 tests: obs shape on reset, step doesn't crash across
all 4 actions, shield intervenes near a hazard, shield leaves a clear-path
action alone, `ShieldedEnv` tracks `_last_obs` without reaching back into
the base env) all pass.

### Open questions / deferred decisions

1. **UE go/no-go** -- measured, but inconclusive (see Goal A above): the
   number obtained isn't the one that determines training speed, and my
   recommended fallback (PyBullet-primary, UE-for-rendering) is a
   recommendation, not a decision made for you. Confirm or override it,
   and say if you want the PIE-based follow-up spike instead.
2. **Episodes essentially never terminate** in the current env/shield
   combination (see regression check above) -- not fixed this phase since
   phase 2 scope explicitly excludes redesigning env/shield logic, but
   Phase 3's shield redesign should account for it.
3. Local-only files `SafeRL_SpaceDebris_Project.md` and
   `saferl_debris_capture/` (present in the working directory from Phase 1
   but never pushed) were left untouched and uncommitted -- out of scope
   for this phase's changes; tell me if you want them added.

### What Phase 3 should assume

- `saferl/` is the source of truth; `2.py`/`Saferl.py` are gone (still in
  git history if needed).
- The PyBullet env (`saferl/env/base_env.py`) trains and passes tests, but
  produces no completed episodes at the current 30k-timestep default --
  don't treat "0 episodes" test output as a new bug introduced by whatever
  Phase 3 changes; it predates this restructure.
- The current `SafetyShield` is still the phase-1 placeholder (random
  action near a hazard) -- real shield logic is explicitly Phase 3/5 scope.
- Default to PyBullet as the training env (this session's recommendation
  pending your sign-off, see Goal A above) with UE reserved for rendering.
  Direct-UE training is not ruled out, but treat it as unvalidated until a
  PIE-based follow-up spike measures real physics-tick throughput -- the
  measurement taken this phase was call-overhead only, not representative
  of training speed.

---

## Phase 3 (2026-09-11): UE Editor + PIE live, episode-termination fix, untracked files landed

### Goal A: UE Editor + PIE running with the environment visible -- achieved

The full Unreal Editor GUI comes up (Vulkan, real rendering -- not headless,
not `-nullrhi`, not a commandlet), spawns the satellite, five debris cubes,
a goal marker and a light, starts a real Play-In-Editor session, and drives
the phase 2 `reset()`/`step()` loop against the PIE-world actor.

Run it with:
```bash
SAFERL_RUN_PIE=1 /mnt/bigdata/unreal_engine/Engine/Binaries/Linux/UnrealEditor \
    ue_spike/SafeRLUESpike.uproject -nosound -log
```
`init_unreal.py` picks up `SAFERL_RUN_PIE=1` and arms
[`ue_spike/Content/Python/pie_session.py`](ue_spike/Content/Python/pie_session.py).
Launched natively, not through Docker -- the engine on `/mnt/bigdata` runs
directly and no container was needed.

**Measured tick-bound rate: ~119 steps/sec** (500 steps in 4.20s;
reproduced at 118.4 / 119.1 / 119.0 across three runs, recorded in
[`ue_spike/pie_session_results.json`](ue_spike/pie_session_results.json)).
The session stayed alive well past the 500-step requirement, and the loop
exercised `reset()` too: the satellite reached the goal and reset 3 times
during the run. This is the number phase 2 could not get -- one env step
per rendered engine frame, capped by the actual frame rate.

Visual record, rendered by the engine itself during the live session:
![live PIE session](ue_spike/pie_session_midrun.png)
([`pie_session_midrun.png`](ue_spike/pie_session_midrun.png),
[`pie_session_final.png`](ue_spike/pie_session_final.png)) -- the large
sphere is the goal marker, the smaller sphere the satellite mid-traverse,
and the five cubes are the debris field. These are `SceneCapture2D` renders
rather than desktop screenshots: this host is Wayland, and X11 screen grabs
of the editor window come back solid black.

**Does this change the phase 2 recommendation? No -- it confirms it, and now
on real evidence rather than a guess.** At 119 steps/sec:

| | 30k steps (current default) | 1M steps |
|---|---|---|
| UE live PIE | ~4.2 min | ~2.3 hours |
| PyBullet (measured, phase 2 + re-run this phase) ~1090/s | ~28 sec | ~15 min |

UE is ~9x slower, which is survivable for a 30k demo run but not for the
step budgets a real safe-RL curriculum wants. So: **keep training in
PyBullet, use UE for rendering and demo playback** -- same recommendation
as phase 2, now measured rather than assumed. Two caveats worth your
judgement before treating 119/s as a hard ceiling: it is one env step per
*rendered* frame, so uncapping the frame rate, running several env steps
per tick, or suppressing rendering during training could each raise it;
and this is still kinematic actor movement, not UE rigid-body physics
under load. **Confirm or override this and I will follow it** -- I have not
changed direction on my own.

Phase 2 got one conclusion wrong and this phase corrects it: the editor
self-terminating after 1-2 ticks had nothing to do with `-nullrhi` "having
nothing to keep it alive". It is `-ExecutePythonScript`, which quits the
editor as soon as the script returns -- it does exactly the same thing in a
full GUI session. Arming from `init_unreal.py` instead keeps the session
alive indefinitely.

Other things worth knowing about running UE this way on this host:
- PIE hangs during audio device init (SDL/pulseaudio); `-nosound` avoids it.
- A `SceneCapture2D` holding a 1920x1080 target re-renders the scene every
  frame and drags the loop from ~119 to ~6 steps/sec. Screenshots are taken
  after the timed run, never during it.
- UE's log buffering stalls under load and makes a perfectly healthy run
  look hung. The session mirrors progress to `ue_spike/pie_heartbeat.log`
  (gitignored) so you can watch it from outside the engine.

### Goal B: episode-termination bug root-caused and fixed

**Root cause was the physics, not the wrappers.** The termination checks in
`env/base_env.py` were correct and nothing was dropping `done` in
`ShieldedEnv` or `Monitor`. The agent simply could not move:
`sphere2.urdf` loads at **10kg with lateralFriction 0.5**, so once it
settles onto the ground plane it sits behind **~49N of static friction**
while the 4-action set can only produce **12N** of thrust. Measured
directly: under constant thrust the agent travels 0.03 units, and its
velocity is then exactly 0.000 for the rest of the run. Neither the goal
check (12.7 units away) nor the collision check could ever fire -- on any
physics backend.

Compounding it, `applyExternalForce` only persists for a single substep, so
one env step held thrust for 1/240s -- about 0.005 m/s of delta-v.

Fixed at the root, both in the env and both config-driven:
- `agent_friction` (default `0.0`) -- a satellite has no ground to rub
  against; the ground plane is an artifact of the toy env.
- `sim_substeps` (default `10`) -- hold thrust across substeps, decoupling
  the ~24Hz control rate from PyBullet's 240Hz physics rate.

`max_episode_steps` (default `1000`) was added **as well**, not instead:
it returns `truncated=True`, never `done=True`, so a wandering policy gets
an episode boundary while genuine goal/collision termination stays
distinguishable from it.

New regression tests in `tests/test_env.py`, all passing (9 total):
- `test_goal_reached_terminates` -- teleport onto the goal, assert `done`
- `test_hazard_collision_terminates` -- teleport onto a hazard, assert
  `done` and `cost == 1`
- `test_agent_actually_moves` -- sustained thrust must displace the agent
  (the direct guard against the friction lock recurring)
- `test_truncation_fires_without_terminating`

**Training regression re-run** (30k timesteps, PyBullet, default config):
**34 completed episodes** and the metrics plot generated, against **0
episodes and no plot** before the fix. `ep_len_mean` settles around 900,
i.e. most episodes still end at the truncation cap rather than on the goal
-- expected for an untrained policy over 30k steps with the shield
deflecting it away from hazards, and worth revisiting when the real shield
lands.

### Goal C: untracked files landed, unmodified

Both were added exactly as they were, at their existing top-level paths,
with no edits and nothing folded into `saferl/`. At a glance:

- **`SafeRL_SpaceDebris_Project.md`** -- 41KB / 1128-line research and
  development plan: "Probabilistic Shield-Augmented Reinforcement Learning
  for Autonomous Capture of Tumbling Space Debris". Abstract, MDP
  formulation, probabilistic shielding theory, tumbling dynamics, and a
  6-phase 9-12 month timeline. Written around **NVIDIA Isaac Lab**.
- **`saferl_debris_capture/`** -- 540KB, 26 tracked files, an 18-module
  Python package scaffold: `envs/` (Isaac Lab env, 36-D obs / 6-D action,
  Euler-equation tumbling dynamics, reward shaping), `shield/` (a **PRISM**
  DTMC model plus abstraction/query/wrapper), `agents/` (PPO, SAC,
  shielded), `training/`, `evaluation/`, `tests/`, and a `docs/paper_draft.md`.
  `__pycache__` excluded by the existing root `.gitignore`.

Flagging without acting on it: both describe a **different simulation stack
(Isaac Lab + PRISM) than the UE + PyBullet line this repo has followed**,
and that `shield/` scaffold is a real probabilistic shield, which is what
phases 3/5 of the current track were meant to build. That is a direction
question for you, not something this phase resolved.

### Open questions / deferred decisions

1. **UE tick-rate recommendation** -- 119 steps/sec confirms PyBullet-for-
   training / UE-for-rendering, but the ceiling may be soft (see caveats
   above). Confirm, or tell me to chase a higher number.
2. **Two parallel project definitions** -- the newly landed Isaac Lab +
   PRISM plan versus this repo's UE + PyBullet track. Which is the real
   roadmap? This affects everything from phase 4 onward.
3. **The shield is still the phase-1 placeholder** (random action near a
   hazard). Untouched this phase, as scoped.
4. Most episodes still end by truncation rather than by reaching the goal
   (`ep_len_mean` ~900 of 1000). Fine for now; revisit with the real shield.

### What Phase 4 (dynamic debris field) should assume

- Episodes genuinely terminate. `done` fires on goal and on collision,
  `truncated` fires at `max_episode_steps`, and there are tests holding
  each of those in place.
- The env is frictionless with a 10-substep control step. Any new dynamics
  work should keep thrust meaningful relative to mass -- the friction-lock
  failure mode is easy to reintroduce and silent when it happens.
- A live UE PIE session is a solved, repeatable path (`SAFERL_RUN_PIE=1`,
  `-nosound`, arm from `init_unreal.py`, never `-ExecutePythonScript`), at
  ~119 steps/sec, driving actors kinematically from Python.
- Training still runs in PyBullet. Do not move training into UE on the
  strength of this phase alone -- open question 1.
- The debris field is static in both backends. The UE scene uses five
  hardcoded debris positions in `pie_session.py`; the PyBullet env still
  randomizes hazard positions per episode with curriculum scaling.

---

## Phase 4 (2026-09-11): Dynamic debris field + UE throughput experiment

### Goal A: dynamic debris field

**Motion model.** Each hazard is assigned a random heading and a speed drawn
from `[debris_min_speed, debris_max_speed]` (default `0.3` - `1.2` env
units/sec) at spawn, then drifts linearly. Position advances by
`velocity * step_dt` each `step()`, where `step_dt = sim_substeps / 240Hz`
(0.0417s with the current config) so debris motion shares the agent's
control timescale rather than running on an unrelated clock. Hazard bodies
are set to **mass 0 and driven kinematically** — otherwise gravity drops
them and contact impulses knock them off their assigned trajectories.

**Boundary behaviour: reflection (bounce).** Debris reflect off the edges of
the `[0, size]` play area, with both position mirrored and velocity
negated. Reasons for picking this over the alternatives: it keeps the debris
count and therefore the observation layout constant (wrapping and
despawn/respawn both do too, but), and unlike wrapping it never teleports an
object across the field into the agent's path. A wrapped object appearing
adjacent to the agent produces an unavoidable collision — no policy and no
shield could have prevented it — which would inject noise into exactly the
safety signal phase 5's shield has to learn from. Despawn/respawn has the
same problem plus a spawn-placement rule to get wrong.

**Observation space: 24 → 39.** Each hazard block is now `[pos(3), vel(3)]`
instead of `[pos(3)]`, so the policy can see motion instead of inferring it.
`OBS_HEADER_LEN` (9) and `OBS_PER_HAZARD` (6) are exported from
`saferl/env/base_env.py` so the layout has exactly one definition;
`SafetyShield` imports them rather than hardcoding a stride. The shield's
*logic* is untouched — still a position-only distance check that picks a
random action — only its indexing moved onto the new stride. Relative
velocity / time-to-collision reasoning is phase 5's job.

**Collision check uses live positions.** `_advance_hazards()` runs before
both `_get_obs()` and the collision check, so both read this step's state.
This is covered by a test that parks a hazard on the agent mid-episode and
asserts the next step terminates — it fails if either path ever goes back to
reading a reset-time snapshot.

**Curriculum on debris speed: implemented, not deferred.** Matching the
existing hazard-count ramp, when `curriculum` is on the *upper* speed bound
ramps from `debris_min_speed` to `debris_max_speed` over
`debris_speed_ramp_episodes` (default 200). Early episodes face near-static
debris; later ones face the full speed band. The lower bound stays fixed so
there is always some motion to react to.

**Tests: 8 new, 17 total, all passing.** Debris actually move between steps
(catches "velocity stored but never applied"), obs tracks live debris
position and velocity, the collision check uses current positions, boundary
reflection keeps every hazard inside the play area across 200 steps, obs
shape matches the new size, PPO builds and predicts against the widened obs
without crashing, and both curriculum speed paths.

**Training regression:** 30k timesteps, no crash, **30 completed episodes**
and the metrics plot written (phase 3's fixed baseline was 34). As expected,
nobody is dodging anything yet — sampling a random policy over 40 episodes
gives 1 goal / 4 crashes / 35 truncations at ~907 mean episode length.
Moving debris does produce real collisions now, which static debris plus the
deflecting shield largely prevented.

### Goal A: shield-relevant content extracted from the Isaac Lab / PRISM files

Read from `saferl_debris_capture/shield/` and `SafeRL_SpaceDebris_Project.md`
directly rather than from phase 3's one-line summary. The stack (Isaac Lab)
is dead per your call, but the **shield design is stack-independent** and
phase 5 should start from it rather than from scratch. What is actually
specified there:

**1. Three-stage architecture.** `abstraction.py` → `shield_query.py` →
`shield_wrapper.py`: map the continuous observation into a small discrete
state, look up a model-checked probability for each (state, action), and
allow-or-override at runtime. Each stage is independently replaceable.

**2. Abstract state `(d, f, t)` — 45 states.** Distance bucket `d ∈ 0..4`
(0 = contact, 4 = far), contact-force bucket `f ∈ 0..2` (0 = safe,
2 = overload), tumble-rate bucket `t ∈ 0..2`. **Translation needed for this
repo:** `f` (contact force) and `t` (tumbling) don't exist in the navigation
task. The natural analogues are distance-to-nearest-debris and a
**closing-rate / time-to-collision bucket** — which is precisely what the
debris velocity added in this phase now makes computable.

**3. Conservative abstraction rule.** When a continuous value sits near a
bucket boundary, round toward the **more dangerous** bucket, so the shield is
never over-optimistic. Cheap to implement, and worth carrying over verbatim.

**4. Safety spec in PCTL, quantitative and checkable:**
`P<=0.05 [ F<=20 "collision" ]` (collision probability within 20 steps ≤ 5%)
and `P>=0.90 [ F<=50 "captured" ]` (task success within 50 steps ≥ 90%).
Having a written, falsifiable spec is the part the current placeholder
shield most obviously lacks.

**5. Override rule: least-restrictive safe action.** If
`P(collision | state, action) > threshold` (default 0.05), substitute the
action that minimises collision probability **while still making progress** —
explicitly not a random action. This is the single biggest delta from this
repo's current shield, which picks uniformly at random from all four
directions and can therefore steer *into* a hazard.

**6. Offline table vs online PRISM.** The design precomputes a
`(num_states, num_actions)` probability table offline and loads it as a
numpy array — zero subprocess overhead per step. Online PRISM invocation is
~100ms/call, which it flags as offline-analysis-only. At ~1090 steps/sec in
PyBullet, anything but a table lookup is a non-starter in the training loop.

**7. Transition probabilities calibrated from rollouts**, not hand-waved:
the `.pm` file's constants are annotated as estimated from offline rollouts
of the dynamics model, with a `shield/estimate_transitions.py` step in the
plan to produce them.

**8. Structured intervention logging** — each intervention records state,
proposed action, substituted action, collision probability and threshold.
This repo currently keeps only an integer counter.

**9. References the design cites:** Jansen et al., *Safe Reinforcement
Learning Using Probabilistic Shields* (CONCUR 2020); Hasanbeig et al.,
*Cautious Reinforcement Learning with Logical Constraints* (AAMAS 2020).

Nothing from these files was implemented this phase, per scope.

### Goal B: UE throughput experiment

Same 500-step protocol as phase 3, four configurations. `pie_session.py`
gained `SAFERL_UNCAP`, `SAFERL_STEPS_PER_TICK` and `SAFERL_RESULTS_NAME`;
defaults reproduce the phase 3 protocol exactly.

| configuration | steps/sec | world ticks for 500 steps | world ticks/sec |
|---|---|---|---|
| capped + throttled, 1 step/tick | 4.2 | 500 | 4.2 |
| **uncapped, 1 step/tick** | **119.0** | 500 | **119.0** |
| uncapped, 10 steps/tick | 1117.3 | 50 | 111.7 |
| uncapped, 50 steps/tick | 4426.3 | 10 | 88.5 |
| *(PyBullet, for reference)* | *~1090* | — | — |

**Two findings, and the second is the one that matters.**

First, phase 3's 119/s was measured with the editor window focused.
Unfocused, the editor throttles to **4.2 steps/sec**. `Slate.AllowThrottling 0`
(alongside `t.MaxFPS 0` and `r.VSync 0`) makes 119/s reproducible regardless
of focus. So 119 was a real number, but it needs those CVars to be dependable
— use `SAFERL_UNCAP=1` for any future measurement.

Second: **the frame-rate ceiling is not soft, and batching does not lift it.**
Look at the last column — world ticks/sec stays flat at 88-119 across every
configuration. Stepping 10 or 50 times per rendered frame does not make UE
simulate faster; it performs more Python-side transform writes *between* the
same ~119 frames. Those extra steps advance no UE physics at all, which is
exactly the caveat that made phase 2's 180,000/s number meaningless.

So: if an env step has to advance UE's own physics — the entire premise of
using UE as the simulator rather than a renderer — **~119 steps/sec is the
real ceiling**, roughly 9x slower than PyBullet, the same ratio phase 3
reported. **The phase 2/3 recommendation stands: train in PyBullet, use UE
for rendering and demo playback.** I did not change the backend, and there is
no new evidence here that would justify revisiting it. Per the time-box, I
stopped after one clean measurement set rather than chasing exotic tuning.

### Open questions / deferred decisions

1. **Debris speed band** (`0.3` - `1.2` env units/sec) was chosen to be
   visibly dynamic without being unavoidable, not calibrated against
   anything physical. If you want Kuiper-belt-realistic relative velocities,
   that is a modelling decision worth making deliberately in phase 5 or 6.
2. **Debris motion is linear, not orbital.** No gravity wells, no relative
   orbital mechanics, and debris pass through each other. Fine for a
   reaction-to-motion task; flagged in case the realism matters later.
3. **The reward is unchanged** and still gives no credit for near-miss
   avoidance, so with moving debris the agent is scored almost entirely on
   goal-reaching and crashes. Worth revisiting alongside the shield.
4. **Untouched from phase 3:** the shield is still the random-action
   placeholder, and the action space is still `Discrete(4)`.

### What Phase 5 (shield redesign) should assume

- **Debris move, and the observation carries their velocity.** Obs is 39-wide
  for `max_hazards=5`: `[agent_pos(3), agent_vel(3), goal_pos(3)]` then
  `[pos(3), vel(3)]` per hazard. Use `OBS_HEADER_LEN` / `OBS_PER_HAZARD` from
  `saferl.env.base_env` rather than hardcoding offsets — changing the layout
  again means changing them in one place.
- **Closing rate is now computable** from the observation, so a
  time-to-collision shield is buildable without touching the env.
- **The shield redesign has a prior design to start from**, summarised above:
  conservative abstraction, a PCTL safety spec, a precomputed probability
  table, and least-restrictive-safe-action overrides instead of random ones.
  The random-action placeholder is actively harmful with moving debris — it
  can steer into a hazard — so replacing the override rule is the highest-value
  single change.
- **Training stays in PyBullet** (~1090 steps/sec), measured again this phase.
  UE remains rendering/demo only, at a confirmed ~119 steps/sec ceiling.
- **Any UE session needs `SAFERL_UNCAP=1`** to avoid the 4.2/s background
  throttle, plus `-nosound` and arming via `init_unreal.py`.
- Episode termination, the friction fix and the truncation backstop from
  phase 3 are all intact and covered by tests (17 passing).

---

## Phase 5 (2026-09-12): Real shield — least-restrictive safe action

Replaces the phase-1 placeholder with an analytic hard-constraint layer. The
soft/learned half of the hybrid design (constrained PPO, cost-value head) is
untouched and remains phase 6's job.

### Goal B first: does the new shield actually avoid collisions the old one caused?

This is the number that matters, so it goes before the design description.

Randomly spawned debris almost never produce a genuine collision course, so
`saferl/eval/stress_test.py` **constructs** them: pick an intercept time,
propagate the agent's un-shielded trajectory analytically
(`p0 + v0·t + ½a·t²`), and place each hazard so that it arrives at that point
at that time. The encounter is then guaranteed and the shield is the only
variable. Six scenarios × 25 trials × three arms:

| scenario | no shield | old (random replacement) | new (least-restrictive) |
|---|---|---|---|
| `closing_head_on` — debris drifting back down the agent's path | 25/25 | 25/25 (24 intervention-implicated) | **0/25** |
| `crossing_from_right` — perpendicular cut across the path | 25/25 | 25/25 (25) | **0/25** |
| `crossing_from_left` — mirror image | 25/25 | 25/25 (25) | **0/25** |
| `parked_obstacle` — stationary, on the path (control case) | 25/25 | 25/25 (25) | **0/25** |
| `surrounded` — four stationary hazards ringing the start | 25/25 | 3/25 (3) | **0/25** |
| `pincer` — two converging flanks plus one ahead | 25/25 | 25/25 (25) | **0/25** |
| **total collisions** | **150/150** | **128/150** | **0/150** |

Mean closest approach to any hazard tells the same story: 1.14–1.29 units
un-shielded, 1.17–1.60 under the old shield, **2.01–2.28 under the new one**
— i.e. the new shield holds the 2.2-unit safe distance it is configured to
hold, while the old one barely improved on having no shield at all.

**The specific flaw being fixed, measured.** "Intervention-implicated" above
counts collisions that occurred within one lookahead window of an
intervention that thrust the agent *toward* the hazard that triggered it.
That audit is pure geometry — the dot product of the substituted thrust
direction with the unit vector from agent to hazard — so it judges both
shields on identical terms rather than through the new shield's own model:

| | old shield | new shield |
|---|---|---|
| mean interventions that thrust toward the triggering hazard | 2.7 – 29.1 per episode | **0** (single-hazard scenarios) |
| collisions implicated by such an intervention | **127 of its 128** | **0** |

So the old shield's collisions were not incidental: essentially every one of
them was preceded by the shield itself steering the agent into the hazard.
That is the concrete, measured version of the flaw phase 4 predicted from the
PRISM design notes.

Two honest caveats. First, in `surrounded` and `pincer` the new shield also
registers "toward-hazard" interventions (30 and 18 per episode) — with debris
on several sides, every direction is toward *something*, so the metric is only
sharp in single-hazard scenarios; its value is that it correlates with
collisions for the old shield and not for the new one. Second, `no_shield`
collides 150/150, which is what makes the comparison meaningful: these
scenarios are lethal by construction, not cherry-picked near-misses.

**Horizon calibration, found the hard way.** At the initially-chosen
`lookahead_steps=20` (0.83 s), the new shield *still* collided 3/3 on
`closing_head_on` and every single intervention was a boxed-in fallback. The
cause is not the selection logic but the horizon: clearing a 2.2-unit safe
radius sideways from rest at `a = force_mag/mass = 1.2 m/s²` needs
`0.5·a·t² ≥ ~1.3`, i.e. **t ≈ 1.5 s ≈ 36 steps**. A shorter horizon only sees
threats that are already unavoidable. At 30 steps it is 0/3 with no
fallbacks; the default is **40 steps (~1.67 s)**, with the derivation written
into `configs/default.yaml` next to the value.

Reproduce with `python -m saferl.eval.stress_test --trials 25`; the table and
raw JSON are checked in at `saferl/eval/stress_results.{txt,json}`.

### Goal A: how the new shield decides

**The trigger now uses relative velocity, not just distance.** For each of
the four actions the shield propagates a forward model over the lookahead
window and rejects any action predicted to come within `safe_dist` of any
hazard:

- agent: constant thrust for the whole horizon, `p(t) = p0 + v0·t + ½a·t²`,
  with `a` read from `ACTION_THRUST_DIRS` in `base_env.py`;
- debris: constant velocity, `h(t) = h0 + hv·t`, read straight out of the
  observation's `[pos(3), vel(3)]` blocks.

Separation is sampled at each of the 40 step boundaries; the action is unsafe
if the minimum falls below `safe_dist`, and its *time-to-violation* is the
first sample time at which it does. `t = 0` is deliberately excluded — the
question is whether this action *leads into* a hazard, not whether the agent
happens to be near one already. A hazard five units away closing at 3 units/s
now triggers, and the identical geometry with the hazard receding does not;
the old distance-only check cannot tell those apart, and a test asserts it
misses the closing case outright.

**"Least different from intent" is measured in thrust vectors, not action
indices.** Action indices are meaningless as a distance metric — 0 and 1 are
adjacent indices but opposite directions (+Y and −Y). The metric used is the
**Euclidean distance between the two actions' commanded thrust vectors**,
which is physical: 0 for the same action, `|a|·√2 ≈ 1.70` for a perpendicular
sidestep, `2|a| = 2.4` for a full reversal. The shield picks the safe action
minimising that distance, so a sidestep always beats a reversal. Ties — a
left and a right sidestep are exactly equidistant from a forward intent — are
broken toward the larger predicted clearance, so the shield dodges away from
the second-nearest hazard rather than into it.

**Boxed-in fallback.** When no action holds `safe_dist` over the horizon, the
shield picks the action maximising time-to-violation (tie-broken on
clearance) — buy time rather than give up. This is tracked distinctly
everywhere:

- `SafetyShield.n_fallback` vs `n_substituted` vs `n_triggered`;
- `ShieldedEnv.fallback_interventions` alongside `interventions`;
- `MetricsCallback.episode_fallback_interventions`, plotted as a second line
  on the interventions panel;
- `ShieldDecision.kind ∈ {"none", "substituted", "fallback"}`.

It is a genuinely different situation — "I chose the least-bad option" rather
than "I found a safe one" — and `surrounded` shows why it still matters: the
fallback fires on all 60 steps of every trial and the agent survives all 25,
while the un-shielded arm dies 25/25.

**Richer intervention record.** Every check produces a `ShieldDecision`
(proposed action, executed action, triggering hazard index, predicted minimum
distance, time-to-violation, thrust deviation, how many safe actions
existed), kept on `shield.last_decision` and optionally appended to
`shield.log` with `keep_log=True`. This is point 8 of the PRISM extraction
from phase 4, and it is what phase 9's evaluation suite will read.

**`ShieldedEnv` did change, slightly.** Its role is unchanged — call the
shield, count, pass through; it makes no safety decisions. The one addition is
the `fallback_interventions` counter, which it fills by reading
`shield.last_decision.kind`. Collapsing fallbacks into the single
intervention count would have discarded exactly the signal phase 9 needs, and
the counter is bookkeeping rather than logic, so it belongs in the wrapper.
`check_and_fix` still returns the same `(action, intervened)` 2-tuple it did
in phase 1, so nothing else needed touching.

**Supporting change in `base_env.py`.** `ACTION_THRUST_DIRS` and `AGENT_MASS`
now live there as the single definition of what an action means; `step()` and
the shield both read them. Previously `step()` hardcoded the four force
vectors, so a shield with its own copy could have silently drifted out of
sync and confidently guarded the wrong world. A test asserts the env's actual
measured acceleration matches the table for every action, and another asserts
`AGENT_MASS` still matches the URDF.

### Goal C: tests and training regression

**Tests: 32 passing** (was 17). `tests/test_shield.py` is new and holds 18;
`tests/test_env.py` keeps the 14 environment tests. The three shield tests
that were in `test_env.py` moved across unchanged in intent — so all 17
phase-4 tests still exist and still pass, they are just split by subject now.
Their observation fixtures were rebuilt on the live 39-wide layout via
`OBS_HEADER_LEN`/`OBS_PER_HAZARD`; the old ones were hand-built on the
pre-phase-4 24-wide stride and only passed by accident.

Covering the four required cases: a hazard closing by velocity alone (with
the receding mirror image, and an assertion that the old shield misses it);
sidestep-preferred-over-reversal with a determinism check across 20 repeated
calls; the boxed-in fallback picking the flee direction that delays the
breach longest; and four no-intervention regressions, including a hazard
tracking alongside the agent at matched velocity — close, but never closing.

**Training regression: 30k timesteps, no crash.** 30 episodes recorded (phase
4's baseline was also 30), 1036 fps — the shield's per-step cost is not
measurable against PyBullet's, since the whole lookahead is one vectorised
`(4 actions × 5 hazards × 40 samples)` numpy pass. Interventions are logged
with the new signal: `30720 checks, 54 triggered (54 substituted, 0 boxed-in
fallbacks)`.

A same-seed 30k run under each shield:

| | episodes | goals | crashes | truncations | triggered |
|---|---|---|---|---|---|
| old random shield | 31 | 0 | 1 | 30 | 200 |
| new shield | 30 | 0 | **0** | 30 | 181 |

As expected and as phase 4 predicted, the policy has not learned to navigate
— every episode still ends at the 1000-step truncation cap, and reward is
flat at −100 (the accumulated step cost). **One crash avoided is a sample of
one** and should not be read as a training-time safety result; the stress
test is where the real evidence is. The point of this run is that nothing
crashed, episodes are still recorded, and the richer intervention signal
flows all the way through to the plot.

### Open questions / deferred decisions

1. **`safe_dist = 2.2` is marginal against the agent's control authority.**
   At 1.2 m/s² the agent needs ~1.5 s to clear that radius sideways, which is
   why the horizon has to be so long. Either number could move: a smaller
   `safe_dist`, a larger `force_mag`, or slower debris would all buy margin.
   I left all three alone because changing them would invalidate the phase
   3/4 baselines mid-comparison — but this is a real tuning decision and
   it is yours.
2. **The forward model ignores boundary reflection.** A hazard predicted to
   fly out of the play area actually bounces. Ignoring it is the conservative
   direction (a bounce can only move a hazard away from its straight-line
   prediction near a wall), so I left it, but it makes the shield slightly
   pessimistic near edges.
3. **The lookahead assumes the action is held for the full horizon**, which
   it is not — the shield re-decides every step. This makes the check
   conservative rather than optimistic, which is the right way to be wrong,
   but it does mean the shield rejects some actions that would in fact have
   been recoverable.
4. **No PCTL spec yet.** Points 2, 3, 4, 6 and 7 of the phase-4 PRISM
   extraction (abstract state buckets, conservative rounding, a written PCTL
   safety spec, an offline probability table, rollout-calibrated transition
   probabilities) are still unimplemented. This phase built the deterministic
   analytic layer only. Whether the probabilistic layer is worth adding on
   top is a real question, not a foregone conclusion — the analytic shield
   already scores 0/150.
5. **The reward still gives no credit for near-miss avoidance**, carried over
   from phase 4 and now more pointed: the shield is doing safety work the
   policy is never rewarded for and cannot see.

### What Phase 6 (constrained PPO / learned soft layer) should assume

- **The hard layer is real and works.** `SafetyShield` enforces a
  velocity-aware constraint and picks the least-restrictive safe action;
  0/150 collisions on scenarios that kill an unshielded agent 150/150. Phase
  6 adds the *soft* layer on top — it does not need to re-derive this one.
- **Interventions carry structure now.** `ShieldDecision` per step,
  `n_triggered`/`n_substituted`/`n_fallback` on the shield,
  `interventions`/`fallback_interventions` on `ShieldedEnv`, and both lists on
  `MetricsCallback`. A cost-value head can be trained against the shield's own
  trigger signal, not just the env's terminal `cost`.
- **`check_and_fix(obs, action) -> (action, intervened)` is stable.** Extra
  signal arrives on `shield.last_decision`, so a Lagrangian wrapper can read
  it without changing the call contract.
- **The action space is still `Discrete(4)`** and the observation still 39-wide.
  Continuous control was explicitly out of scope this phase; the shield's
  selection loop enumerates actions, so it would need a different formulation
  (a projection or a QP) to go continuous.
- **The policy still does not reach the goal within 30k steps.** Reward is
  flat at −100 with every episode truncating. Phase 6 should expect to fix
  learning, not just safety — and the shield now guarantees the exploration
  it does is not fatal.
- Training stays in PyBullet (~1036 fps measured again this phase, unchanged
  by the shield). UE remains rendering/demo only.

---

## Phase 6 (2026-09-12): Margin retune + constrained PPO (learned soft layer)

Goal A loosened the physics so a policy has room to work inside the shield's
margin. Goal B added the soft layer: PPO is now penalised for *needing* the
shield, not just protected by it.

### Goal A: margin retune

**`force_mag` 12 N → 48 N** (1.2 → 4.8 m/s² against the agent's 10 kg).
Dodging a hazard means displacing sideways by `hazard_threshold` (1.3) before
it arrives, so `0.5·a·t² ≥ 1.3` gives the reaction time. That is `t ∝ 1/√a`,
so **4× the thrust halves the clearing time**: 1.47 s (~36 steps) → **0.74 s
(~18 steps)**, inside the 15–20 step target. Clearing the full 2.2-unit
`safe_dist` rather than the collision radius takes 0.96 s (~23 steps); both
numbers are quoted because phase 5's "~36 steps" was the 1.3 figure and the
comparison should be like-for-like.

**`lookahead_steps` 40 → 20.** The horizon is what bounds the shield's
reachable set, `0.5·a·(H·dt)²`. At 4× thrust and half the horizon that set is
`0.5·4.8·0.833² = 1.67` units — *exactly* what phase 5 had at `H=40` and
1.2 m/s². So the shield sees the same distance ahead of itself while doing
half the arithmetic per check. Verified by sweep rather than assumed
(6 scenarios × 5 trials):

| horizon H | collisions | fallback interventions |
|---|---|---|
| 8 | 5/30 | 181 |
| 12 | 0/30 | 60 |
| 16 | 0/30 | 60 |
| **20 (chosen)** | **0/30** | **60** |
| 24 / 28 / 40 | 0/30 | 60 |

The empirical floor is H=12; H=20 keeps a margin and sits where intervention
count plateaus (114 at H=20 vs 107 from H=24 on). All 60 remaining fallbacks
come from `surrounded`, which is boxed-in by construction at every horizon.

### Fresh stress baseline under the new physics

Re-run, **not** carried over from phase 5 — these numbers describe the current
config only (6 scenarios × 25 trials):

| arm | collisions | preceded by a toward-hazard intervention |
|---|---|---|
| no shield | 150/150 | — |
| old random-replacement shield | **146/150** | 135 |
| new least-restrictive shield | **0/150** | 0 |

The old shield got *worse* than phase 5's 128/150. That is the expected
direction: more thrust means a wrong random dodge covers more ground, so the
placeholder's failure mode is amplified by exactly the change that helps the
real shield. The new shield stays at zero.

**Three things the physics change broke, all found by measurement:**

1. **The stress scenarios silently stopped testing anything.** They specified
   encounters in *seconds*, so at 4× thrust every intercept point moved
   outside the 10-unit play area — where debris reflect off the boundary and
   leave their designed course. `pincer` degraded to 5/5 collisions with
   *zero* interventions and would have been reported as a shield regression.
   Encounters are now specified as **distance along the path** with the
   harness solving for the time, and a guard raises if an encounter would land
   outside the field, so this cannot pass quietly again.
2. **`tests/test_shield.py` hardcoded `lookahead_steps=40` and
   `safe_dist=2.2`**, combining the new thrust with the old horizon to give
   the shield a 6.67-unit reachable set. One test failed and the rest were
   passing for the wrong reason. Tests now read `configs/default.yaml`,
   `SafetyShield`'s module defaults are *derived* from it rather than
   restated, and `SafetyShield.from_config()` exists for non-default configs.
   The same sweep found and fixed stale literals in `SafeNav3DEnv.__init__`,
   `stress_test.run_comparison` and `test_env.py`.
3. **The agent could leave the universe.** Nothing walls it in and there is no
   drag, so at 4.8 m/s² a wandering policy reached `|xy| ≈ 158` within one
   episode — outside the observation space's own ±20 bounds, with hundreds of
   steps spent where the goal is unreachable and no hazard is in view. See
   below; this turned out to matter far more than it first looked.

### The degenerate escape, and why it had to be fixed first

With leaving the field unpenalised, PPO found a much better idea than doing
the task: **fly out of bounds as fast as possible.** Over 300k steps it
converged by ~70k and sat there — 0% goals, 0% collisions, ~36-step episodes,
task reward flat at −3.6. The arithmetic is not subtle:

| behaviour | return |
|---|---|
| escape immediately (~36 steps) | **−3.6** |
| wander in-field for a full episode | −100 |
| reach the goal (~80 steps) | +92 |
| crash | ≈ −50 |

Escape dominates everything the policy had actually found, and the +92 the
task offers is behind an exploration barrier. Worse, this **starved the soft
layer of any signal**: over the same 300k steps the constrained arm's λ
saturated at its ceiling of 50 and the intervention rate still went *up*
(0.0110 → 0.0120). An agent beelining out of the field has almost no control
over whether debris happen to sit on its exit path, so no multiplier, however
large, can change its behaviour.

**Fix: out-of-bounds terminates with a penalty.** The value is derived
rather than fitted — **−100 equals a full episode's accumulated step cost**
(`max_episode_steps × 0.1`), which is exactly the threshold above which
leaving early stops being better than staying and trying. Swept at 120k steps
to confirm the reasoning survives contact:

| `out_of_bounds_penalty` | goal % | escape % | mean steps | intervention rate |
|---|---|---|---|---|
| 0 | 0.0 | 100.0 | 37 | 0.010 |
| −25 | 40.7 | 50.0 | 460 | 0.244 |
| −50 | 0.0 | 5.9 | 976 | 0.079 |
| **−100 (chosen)** | **42.2** | **20.0** | 659 | 0.224 |

−100 gives the best goal rate and the least escaping. −50 is a trap: the agent
learns to hover in-field for the full 1000 steps and never commits.

### Goal B: the cost signal and the Lagrangian layer

**Cost signal.** `ShieldedEnv` now emits `info["shield_cost"]` — 1 on any step
the shield intervened (substitution *or* fallback), 0 otherwise — plus
`info["shield_fallback"]`. This is deliberately distinct from the env's
existing collision `cost`: *needing the shield* is what the policy should
learn to stop doing, *crashing* is what the shield exists to prevent.

**The multiplier is real; its application is simplified.** Being precise about
which half is which:

- **Real:** λ is updated by a genuine dual-gradient ascent step on the
  constraint violation, at episode boundaries:
  `λ ← clip(λ + lr·(ema_rate − target_rate), 0, λ_max)`.
  It rises while the policy exceeds its budget and decays once under. Not a
  schedule, not a fixed penalty.
- **Simplified:** λ is applied by **shaping the scalar reward**
  (`r' = r − λ·cost`) and learned through PPO's single existing critic. A
  textbook PPO-Lagrangian trains a **separate cost-value head** and forms the
  policy gradient from both critics. This is the same dual variable with one
  critic instead of two. The brief allowed either; this is the simpler one and
  is labelled as such.

The constraint is on intervention **rate**, not the episode total, because the
out-of-bounds backstop makes episode length vary ~4×, and a per-episode budget
would mostly reward ending episodes early.

`enabled=False` pins λ at 0, so the unconstrained baseline runs through
byte-identical plumbing and the comparison isn't confounded by a different
code path.

### CPU vs GPU: measured, and CPU won

| device | 20k timesteps | throughput |
|---|---|---|
| **CPU (now the default)** | **23.89 s** | **837 fps** |
| CUDA | 33.80 s | 592 fps |
| *env + shield alone, no network* | *7.41 s* | *2700 fps* |

Same seed, best of two reps each. **CPU is 1.41× faster.** A 64×64 `MlpPolicy`
has nowhere near enough arithmetic per batch to amortise host↔device transfer.
`training.device: cpu` is now the configured default.

### Constrained-vs-unconstrained training results (400k steps)

Both arms ran for 400k timesteps under identical physics, shield, and OOB
penalty. The only difference is whether λ can rise above 0.

**Unconstrained arm** (891 episodes):

| decile (timestep) | task reward | intervention rate | ivs/episode | steps | goal % |
|---|---|---|---|---|---|
| 38k  | −109.9 | 0.041 |  23.6 | 436 |  10.1 |
| 89k  |  −56.4 | 0.165 |  93.9 | 564 |  36.0 |
| 192k |  −31.1 | 0.205 | 117.9 | 513 |  48.3 |
| 268k |  +46.5 | 0.180 |  89.4 | 411 |  91.0 |
| 401k |  +61.8 | 0.194 |  69.3 | 327 |  95.6 |

The agent learned to reach the goal reliably (95.6% in the final decile) —
the **first time in this project any policy has done so**. Task reward rose
from −110 to +62. Intervention rate settled at ~19%, meaning on roughly 1 in 5
steps the shield had to correct the agent's action. Zero collisions throughout.

**Constrained arm** (509 episodes):

| decile (timestep) | task reward | intervention rate | ivs/episode | steps | goal % | λ |
|---|---|---|---|---|---|---|
| 14k  | −107.3 | 0.024 |   7.5 | 273  | 10.0 | 0.000 |
| 56k  | −110.2 | 0.066 |  52.5 | 827  |  2.0 | 0.013 |
| 129k | −107.1 | 0.095 |  76.8 | 679  |  5.9 | 0.119 |
| 213k | −117.8 | 0.048 |  40.9 | 864  |  0.0 | 0.504 |
| 303k | −112.1 | 0.007 |   6.8 | 906  |  0.0 | 0.261 |
| 401k | −100.9 | 0.015 |  14.5 | 990  |  0.0 | 0.000 |

The constraint worked exactly as intended: intervention rate dropped from 4.6%
to 1.5%, well below the 5% target. λ peaked at ~0.5, then decayed to 0 as the
rate fell under the target. But the **cost was catastrophic**: goal rate
collapsed to 0% in the last decile, task reward stayed at −101, and episode
length ballooned to ~990 steps (nearly the 1000-step cap). The agent learned
to be extremely cautious — it stopped triggering the shield by stopping doing
anything at all.

**Final-decile comparison:**

| metric | unconstrained | constrained | delta |
|---|---|---|---|
| interventions/step | 0.194 | 0.015 | −92.5% |
| interventions/episode | 69.3 | 14.5 | −79.0% |
| task reward | +61.8 | −100.9 | −162.7 |
| goal rate % | 95.6 | 0.0 | −95.6 pp |
| collision rate % | 0.0 | 0.0 | +0.0 |
| episode length | 327 | 990 | +663 |

This is the classic safety-performance tradeoff with a simplified Lagrangian:
a single critic cannot simultaneously track task value and cost value, so the
policy converges to the constraint-satisfying fixed point closest to its
initialisation — which is "do nothing" rather than "navigate efficiently while
avoiding hazards." The dual variable did its job (rose while the rate was over
budget, fell when it dropped below), but the policy did not have the
representational or learning capacity to satisfy both objectives at once.

### Sanity regression (30k steps, final config)

Standard training pipeline under the final config confirms nothing is broken:
87 episodes, 1339/30720 shield triggers (4.4% rate), 17 fallbacks, 0
collisions. Reward and episode length profiles match pre-phase-6 expectations
at this step count.

### Test suite

All 36 tests pass: 18 in `test_env.py` (including 4 new out-of-bounds tests)
and 18 in `test_shield.py`. The 4 new tests verify OOB termination, penalty
magnitude, margin tolerance, and goal-termination priority.

### Open questions / what Phase 7 should assume

1. **The safety-performance tradeoff is real and unsolved.** The single-critic
   Lagrangian reduced interventions at the cost of all task performance. To fix
   this, Phase 7 should consider:
   - **Separate cost-value head** (the textbook PPO-Lagrangian): lets the
     policy gradient balance task reward and cost penalty without one critic
     trying to regress both signals.
   - **Curriculum on the constraint**: start with a generous target rate (e.g.
     0.30) and anneal toward 0.05, so the policy first learns *how* to reach
     the goal, then learns to do so without the shield.
   - **Reward shaping for near-miss avoidance**: the policy gets no credit for
     dodging a hazard cleanly (phase 5 open question #5, still unanswered).
     Shield-proximity reward could help the constrained policy find the
     navigate-and-dodge behaviour rather than the do-nothing behaviour.
2. **The unconstrained policy reaches goals at 95.6% — this is the project's
   first working policy.** Phase 7 can use it as a performance baseline and a
   warm-start for constrained fine-tuning, which would bypass the cold-start
   exploration problem that trapped the constrained arm.
3. **CPU is faster than GPU for this network size.** This will change if the
   policy grows (larger network, attention, etc.) or if envs are vectorised.
4. **`SafetyShield.from_config(cfg)` is the stable construction API.** New
   code should use it rather than positional arguments.
5. **The OOB penalty is load-bearing** — without it, neither arm learns
   anything useful. It must be preserved in all future configs.

---

## Phase 6b (2026-09-12): Fix constrained PPO collapse (redo of Goal B)

Phase 6's constrained arm collapsed to 0% goal rate because a single critic
conflated task reward and cost penalty — the policy satisfied the intervention
constraint by freezing rather than navigating. This phase fixes the
architecture with three complementary measures: a separate cost-value head,
a curriculum on the constraint target, and warm-starting from the working
unconstrained policy. Named "6b" rather than "7" because this resolves an
open problem from phase 6, not new scope.

### The two-critic architecture

SB3's `ActorCriticPolicy` was subclassed as `DualCriticPolicy` with a second
value head (`cost_value_net`), a `Linear(64, 1)` sharing the same critic
feature extractor as the task-value head but with independent weights. SB3
does not have first-party support for two-critic constrained PPO, so this
required three custom classes:

- **`DualCriticPolicy`**: adds `cost_value_net` alongside the standard
  `value_net`. `forward_all(obs, actions)` returns `(values, cost_values,
  log_prob, entropy)` in one forward pass. The actor network is unmodified.
- **`CostRolloutBuffer`**: extends `RolloutBuffer` with parallel cost arrays
  (`cost_rewards`, `cost_values`, `cost_returns`, `cost_advantages`). Computes
  cost-specific GAE alongside standard task GAE. Combined advantages
  `A_task - λ·A_cost` are set before the training loop.
- **`ConstrainedPPO`**: subclasses `PPO` to override `collect_rollouts()`
  (collects cost signals and predicts cost values) and `train()` (adds cost
  value loss and uses combined advantages for the clipped surrogate).

The Lagrange multiplier update uses the same dual-gradient formula as phase 6
(`λ ← clip(λ + lr·(ema_rate − target), 0, max)`) but now drives the policy
gradient through cost advantages rather than reward shaping. This is the
architectural fix: the task critic sees only task reward, the cost critic sees
only intervention cost, and the policy gradient balances both instead of one
critic trying to regress both signals.

### Curriculum on the constraint target

The constraint target ramps linearly over training:

| training progress | target rate | rationale |
|---|---|---|
| 0% (start) | **0.25** | Above the unconstrained policy's ~0.19 natural rate |
| 50% | 0.15 | Gradually tightening |
| 100% (end) | **0.05** | Phase 6's final target (retained) |

This prevents the cold-start collapse: early in training, the constraint is
non-binding (target > natural rate), so the policy preserves its navigation
skill. As the target tightens, the policy adapts to reduce interventions while
maintaining task performance. The schedule is tied to `_current_progress_remaining`
(same mechanism SB3 uses for learning rate schedules), consistent with phase 4's
debris-speed curriculum pattern.

### Warm-start verification

The unconstrained policy's 400k-step checkpoint (`saferl_unconstrained.zip`)
was loaded into `DualCriticPolicy`. 12 of 14 parameters matched exactly
(all actor and task-critic weights); the 2 new `cost_value_net` parameters
were randomly initialized.

**Verification**: action distributions were compared between the original PPO
model and the warm-started `DualCriticPolicy` on identical observations —
probabilities matched to floating-point precision (`[0.9952, 0.000006,
0.000346, 0.00448]`). The warm-started model evaluated at **~60% goal rate**
on 300 fresh episodes (stochastic policy), consistent with the original
model's overall training rate of 66.3% (the 95.6% reported in phase 6 was
the last-decile in-distribution metric, not a generalization number).

### Training results (400k steps)

| decile | timestep | goal % | iv rate | task reward | lambda | target |
|---|---|---|---|---|---|---|
| D1 | 53k | 26 | 0.198 | -60.1 | 0.000 | 0.224 |
| D2 | 115k | 85 | 0.155 | +40.3 | 0.000 | 0.193 |
| D3 | 147k | 81 | 0.173 | +35.4 | 0.000 | 0.177 |
| D4 | 175k | 76 | 0.179 | +27.5 | 0.003 | 0.163 |
| D5 | 209k | 84 | 0.186 | +37.5 | 0.039 | 0.146 |
| D6 | 243k | 82 | 0.164 | +35.8 | 0.118 | 0.129 |
| D7 | 276k | 89 | 0.181 | +52.0 | 0.193 | 0.113 |
| D8 | 312k | 89 | 0.192 | +50.8 | 0.328 | 0.094 |
| D9 | 348k | 88 | 0.196 | +49.4 | 0.506 | 0.076 |
| D10 | 385k | **88** | **0.138** | **+48.5** | 0.700 | 0.058 |

1100 episodes total. 868 goals (78.9%). **Zero collisions.**

**Final-decile comparison across all three runs:**

| metric | unconstrained (ph6) | constrained (ph6) | phase 6b |
|---|---|---|---|
| goal rate % | 95.6 | 0.0 | **88.2** |
| intervention rate | 0.194 | 0.015 | **0.138** |
| task reward | +61.8 | -100.9 | **+48.5** |
| lambda | 0.0 | 0.0 | 0.700 |
| collisions | 0 | 0 | 0 |

### Assessment: partially resolved

**What worked.** The two-critic architecture definitively prevents the phase-6
collapse. The policy maintained 88% goal rate under active constraint pressure
(lambda=0.700 and rising), compared to 0% in phase 6's single-critic
constrained arm. The cost critic's loss decreased from ~8 to ~4 over training,
confirming it learned the cost signal. The warm-start and curriculum both
functioned as intended — the policy never fell into the freezing attractor.

**What remains.** The intervention rate decreased 29% from the unconstrained
baseline (0.138 vs 0.194) but did not reach the 5% target. Lambda was still
rising at the end of training, indicating the dual-gradient ascent hasn't
converged. The policy found a regime where it navigates well (88% goals) with
moderate cost (~14% of steps trigger the shield) while lambda applies growing
pressure — but 400k steps wasn't enough for that pressure to substantially
change the navigation strategy.

This is an honest tradeoff, not a failure mode: the policy IS navigating and
IS under constraint pressure and IS reducing interventions. It just hasn't
reduced them *enough*. In phase 6's collapsed run, the policy trivially
satisfied the constraint by not navigating — that pathology is gone.

### Test suite

All 36 tests pass (18 in `test_env.py`, 18 in `test_shield.py`). No changes
were made to `env/base_env.py` or `shield/safety_shield.py`.

### What Phase 7 should assume

1. **The hard shield still works.** 0/150 collisions (stress test), 0 training
   collisions across 1100 episodes. The shield config (`force_mag=48`,
   `lookahead_steps=20`, `safe_dist=2.2`) is unchanged from phase 6.
2. **The two-critic constrained PPO is the correct architecture.** Use
   `ConstrainedPPO` + `DualCriticPolicy` from `dual_critic.py`. The old
   single-critic `CostPenaltyWrapper` in `constrained.py` is retained for
   reference but should not be used for new training.
3. **The unconstrained policy (~60% eval goal rate) is the warm-start source.**
   `saferl/eval/phase6/saferl_unconstrained.zip` weights load cleanly into
   `DualCriticPolicy` — 12/14 params, cost head initialised fresh.
4. **To push the intervention rate lower**, the most promising levers are:
   - Longer training (lambda was still rising at 400k)
   - Lower curriculum start (tighter constraint earlier, once warm-start
     policy is stable)
   - Reward shaping for near-miss avoidance (still the most-deferred open
     question — the policy gets no credit for cleanly dodging a hazard)
   - Warm-starting from phase 6b's checkpoint (which already has some
     constraint adaptation) rather than from the unconstrained one
5. **CPU remains faster than GPU** for this network size and env count.


---


## Phase 7 — Limited-Range Sensing

**Goal:** Make the policy's observation realistic — it should only see hazards
within `sensor_range`, while the safety shield retains privileged access to
all true hazard state.

### Design decision

The safety shield keeps its full, privileged view of true hazard state. It is
*not* limited to the policy's sensor range. This is a deliberate choice: the
shield models a dedicated collision-avoidance system (like TCAS in aviation)
with its own sensor path, not a software layer that must share the main
controller's perception. The policy sees what a limited sensor would see; the
shield sees what a safety-critical backup system would see.

### Implementation

1. **Config** (`default.yaml`): `sensor_range: 6.0` — hazards beyond 6 units
   from the agent appear as zeros in the policy's observation. `null` disables
   the limit (god's-eye view, the pre-phase-7 default).

2. **Observation split** (`base_env.py`): `_get_obs()` refactored into
   `_build_obs(sensor_limited)`. `_get_obs()` calls it with `True` (policy
   view); `get_true_obs()` calls it with `False` (shield view). The zero
   padding uses the same convention as curriculum-unused hazard slots — no
   observation-space change needed.

3. **Privileged shield** (`safety_shield.py`): `ShieldedEnv` tracks
   `_last_true_obs` alongside `_last_obs`. The shield receives true obs via
   `check_and_fix(self._last_true_obs, action)`, while the policy receives
   sensor-limited obs from `step()` and `reset()`.

### Training results

Fine-tuned from the phase 6b constrained checkpoint (200k steps, seed 17):

| Metric | Phase 6b (baseline) | Phase 7 (limited sensing) |
|--------|--------------------:|-------------------------:|
| Goal rate (eval, 50 ep) | ~60% (stochastic) | **86.0%** |
| Goal rate (training) | 78.9% | **83.8%** |
| Intervention rate (eval) | ~18% | **7.46%** |
| Intervention rate (last 20%) | 16.7% | **11.4%** |
| Collisions | 0 | **0** |
| Lambda (final) | 0.77 | **1.70** |
| Episodes | 1100 | 579 |

The policy adapted well to partial observability. The higher lambda (1.70
vs 0.77) reflects the tighter constraint being enforced under harder conditions.

**Key observation:** the privileged shield compensates for the policy's blind
spots. When a hazard approaches from outside sensor range, the shield
intervenes even though the policy has no information about that hazard. The
policy then learns to spend less time in geometries where the shield must
intervene — even though it cannot directly observe what triggers the shield.

### Tests (40/40 passing)

Four new sensor-range tests:
- `test_out_of_range_hazard_zeroed_in_obs` — hazard beyond range → zeros
- `test_in_range_hazard_appears_in_obs` — hazard within range → real data
- `test_true_obs_shows_all_hazards_regardless_of_range` — get_true_obs() bypass
- `test_shield_intervenes_on_out_of_sensor_range_hazard` — **the core proof**:
  hazard outside sensor range on collision course, policy obs shows zeros, but
  the shield sees it via true obs and intervenes

### Stretch goals — deferred

- **Sensor-range curriculum** (gradually shrinking range during training):
  not implemented. The policy achieved 86% goal rate with a fixed 6.0 range
  from the warm-start — no evidence a curriculum would help.

- **Sensor noise** (Gaussian noise on in-range hazard observations):
  deferred to a future phase. The core limited-sensing signal is clean; noise
  adds realism but would obscure whether any degradation comes from range
  limits or from noise.

### Files

- `saferl/training/train_phase7.py` — fine-tuning script
- `saferl/eval/phase7/` — training CSV, plot, checkpoint, summary
- `saferl/configs/default.yaml` — `sensor_range: 6.0`
- `saferl/env/base_env.py` — `_build_obs()`, `get_true_obs()`
- `saferl/shield/safety_shield.py` — `ShieldedEnv._last_true_obs`


---


## Phase 8 — Eval Reconciliation + Visual Fidelity (2026-09-12)

### Goal A: Phase 6b vs. Phase 7 baseline reconciliation

Phase 6b's README section reported the accepted final result as **88.2% goal
rate, 13.8% intervention rate**. Phase 7's README section cited the phase 6b
baseline as **78.9% goal rate, ~18% intervention rate**. These are different
numbers for the same checkpoint. This phase resolves the discrepancy.

#### Root cause

Both numbers come from the **same checkpoint** (`saferl/eval/phase6b/saferl_phase6b.zip`,
md5 `60bc46a8a1207c8d8e86b5f47043b482`). The discrepancy is not about different
weights — it is about different slices of the same training run's episode data:

- **88.2% / 13.8%** = the **last decile** (final 110 of 1100 training episodes).
  This is the peak performance the policy reached by end of training.
- **78.9% / ~18%** = the **overall training average** across all 1100 episodes,
  including the early learning curve where the policy was still adapting from
  the warm-start. The ~18% intervention rate is the last-20% average (0.167).

Neither number was from a proper held-out evaluation. The 88.2% overstates the
checkpoint's true capability (it cherry-picks the best training window), while
78.9% understates it (it averages in the early learning phase). The phase 7
comparison table mixed these metric types without flagging the difference.

#### Eval methodology audit

| Detail | Phase 6b's 88.2%/13.8% | Phase 7's citation of 78.9%/~18% |
|--------|------------------------|----------------------------------|
| Source | Training callback log | Training callback log |
| Window | Last decile (110 ep) | Full run (1100 ep) / last 20% |
| Action selection | Stochastic (training) | Stochastic (training) |
| Eval episodes | N/A (training data) | N/A (training data) |
| Seed | Training seed 17 | Training seed 17 |
| Held-out eval? | No | No |

Both were training-time metrics, not evaluations. Phase 7 also ran a
`quick_eval` (100 episodes, stochastic) that produced ~60% goal rate /
~18% intervention, but this used `deterministic=False` and only 100 episodes.

#### Authoritative re-evaluation

A new standardized evaluation protocol (`saferl/eval/evaluate.py`) was created
for phases 8+:

- **500 episodes** (sufficient to bound a 10-point gap at 95% confidence)
- **Deterministic action selection** (policy mean, not sampled)
- **Fixed seed (42)** for reproducibility
- **Per-episode CSV** output for downstream analysis
- **Checkpoint MD5** recorded for traceability

Results on the phase 6b checkpoint (500 ep, deterministic, seed 42):

| Condition | Goal rate | Intervention rate | Collisions |
|-----------|-----------|-------------------|------------|
| Phase 6b, full sensing (range=100) | **71.6%** (358/500) | 18.47% | 0 |
| Phase 6b, limited sensing (range=6.0) | **66.6%** (333/500) | 28.82% | 0 |
| Phase 7, limited sensing (range=6.0) | **89.2%** (446/500) | 12.57% | 0 |

#### Corrected phase 7 comparison

Against the authoritative phase 6b baseline under matching conditions
(limited sensing, same eval protocol):

| Metric | Phase 6b baseline | Phase 7 | Change |
|--------|------------------:|--------:|-------:|
| Goal rate | 66.6% | **89.2%** | **+22.6 pp** |
| Intervention rate | 28.82% | **12.57%** | **-16.25 pp** |
| Collisions | 0 | 0 | — |

**Phase 7's improvement is real and larger than originally reported.** Under
equal conditions, limited-range sensing training improved goal rate by 22.6
percentage points and cut the intervention rate by more than half. The original
comparison understated the improvement because it compared phase 7's training
average against phase 6b's training average, both of which were diluted by
their respective learning curves.

The likely mechanism: the sensor-limited policy learned to navigate more
conservatively (avoiding geometries where unseen hazards could appear), which
simultaneously improved goal-reaching reliability and reduced shield
interventions — a case where the constraint and the objective aligned rather
than traded off.

### Standardized eval protocol (phases 8+)

All future phase evaluations should use `saferl/eval/evaluate.py`:

```bash
python -m saferl.eval.evaluate --checkpoint <path> [--episodes 500] [--seed 42]
```

- Default: 500 episodes, deterministic, seed 42
- `--stochastic` for stochastic evaluation
- `--sensor-range <float>` to override the config's sensor range
- `--out-dir <path>` to save per-episode CSV and summary CSV
- Reports checkpoint MD5 for traceability

This replaces the ad hoc `quick_eval()` functions in individual training
scripts for reporting purposes. Training scripts may still use their own
eval for progress monitoring, but authoritative cross-phase comparisons
must use this protocol.

### Goal B: Visual fidelity

The RL/safety core has been the focus through phase 7. This phase begins the
visual work in Unreal Engine — replacing the placeholder primitives from
phase 3 with geometry that reads as a satellite navigating among space debris.

#### Skybox / space environment

Default UE sky atmosphere, sky light, exponential height fog, and volumetric
clouds are removed at scene-build time. A large inverted-normal sphere
(`sky_dome.obj`, 642 verts, 1280 faces) provides a dark backdrop. The OBJ
is imported via `AssetImportTask` at first run.

#### Satellite mesh

The placeholder sphere is replaced with a **composite actor** built from
basic shapes:
- **Body:** Cube scaled to a rectangular box (1.2 x 0.8 x 0.6)
- **Solar panels:** Two thin cubes (0.05 x 1.6 x 0.5) offset on opposite sides
- **Antenna boom:** Thin cylinder (0.1 x 0.1 x 1.2) rising from the body
- **Dish:** Flattened sphere (0.3 x 0.3 x 0.15) at the antenna tip

All parts are attached to the body actor via `attach_to_actor()`, so the
existing `step()` loop moves them as one unit by repositioning the body alone.

**Source/license:** All meshes are UE engine built-in basic shapes
(`/Engine/BasicShapes/`), shipped with every UE installation. No external
assets or licenses required.

#### Debris meshes

The 5 placeholder cubes are replaced with **5 unique irregular rock meshes**
generated by `ue_spike/Content/Python/generate_meshes.py`:
- Each is a perturbed icosphere (162 verts, 320 faces) with a unique random
  seed controlling the vertex displacement
- Non-uniform per-axis scale (0.5–1.5) creates elongated/flattened shapes
- 35% vertex noise strength produces genuinely irregular, rocky geometry
- Each debris actor has a unique rotation and non-uniform world scale
  (varied per-axis between 1.3 and 2.2)

**Source/license:** Procedurally generated in pure Python (no external
libraries). The generator script and all 5 OBJ files are committed to the
repository. No external assets or licenses required.

#### Lighting

- Single `DirectionalLight` at 5800K color temperature, intensity 8.0
- Pitched at -30 degrees, rotated to cast shadows across the debris field
- Shadows enabled, no atmospheric scattering
- Consistent with harsh, single-source illumination in deep space

#### Control loop verification

The existing `PIEBridge.reset()`/`step()` control loop was reviewed for
compatibility with the new meshes:

- The satellite body actor (which the step loop repositions) retains its
  `SATELLITE_TAG`, so `_find_satellite_in_pie()` locates it correctly in PIE
- Solar panels, antenna, and dish are attached via `attach_to_actor()` with
  `KEEP_WORLD` attachment rules, so they move with the body automatically
- Debris actors are spawned as `StaticMeshActor`s with `MOVABLE` mobility,
  same as before — `set_actor_location()`/`set_actor_rotation()` work
  identically regardless of mesh geometry
- The step loop only calls `set_actor_location()` on the body actor, which
  does not depend on mesh pivot point — UE basic shapes and imported OBJs
  both default to origin-centered pivots

**Risk note:** If the imported OBJ meshes have unexpected pivot offsets (the
generator centers them at origin, but UE's import pipeline may shift them),
the visual position could be offset from the expected env-space coordinate.
This would be a visual-only issue (the physics/RL simulation runs in PyBullet,
not UE), but should be checked on first PIE run and corrected via the import
settings if needed.

#### Screenshot

The PIE session captures screenshots automatically via `SceneCapture2D` at the
end of the step loop run (same mechanism as phase 3). The screenshots from the
next PIE run with the visual fidelity changes will be saved to
`ue_spike/pie_session_midrun.png` and `ue_spike/pie_session_final.png`,
replacing the phase 3 placeholder-geometry captures.

### Tests

All **40 tests pass** (18 in `test_env.py`, 22 in `test_shield.py`). No
changes were made to any RL, physics, or shield code. The visual changes are
entirely within `ue_spike/` and do not affect the training/evaluation pipeline.

### Files

- `saferl/eval/evaluate.py` — standardized evaluation protocol
- `saferl/eval/phase8_reconciliation/` — authoritative eval results (CSVs)
- `ue_spike/Content/Python/generate_meshes.py` — procedural mesh generator
- `ue_spike/Content/Python/pie_session.py` — updated scene builder
- `ue_spike/Content/Meshes/debris_rock_0..4.obj` — 5 unique rock meshes
- `ue_spike/Content/Meshes/sky_dome.obj` — inverted sky sphere

### What Phase 9 should assume

1. **The authoritative phase 6b baseline is 71.6% goal / 18.47% intervention**
   (500 ep, deterministic, seed 42, full sensing). Use `saferl/eval/evaluate.py`
   for all future cross-phase comparisons.
2. **Phase 7's improvement is confirmed:** 89.2% goal / 12.57% intervention
   under limited sensing, a +22.6pp / -16.25pp improvement over the corrected
   baseline.
3. **The eval protocol is standardized.** Future phases cite the protocol
   rather than restating methodology.
4. **The UE visual environment is ready** for rendering/demo use. The PIE
   session shows a composite satellite among irregular debris in a dark space
   environment with harsh directional lighting.
5. **RL code is untouched.** All 40 tests pass. The training pipeline,
   shield, and env are unchanged from phase 7.
6. **Phase 9 scope:** full training run with the phase 7 architecture,
   evaluation suite using the standardized protocol, metrics dashboard.

---

## Phase 9 (2026-09-12): Long training run + TensorBoard dashboard

Extended the constrained training run to convergence under plateau detection,
with live TensorBoard metrics. Discovered that with the phase 7 architecture,
the policy oscillates rather than monotonically improves, while the Lagrange
multiplier climbs without bound — the signature of an unreachable constraint target.

### Goal A: Convergence-based long training run

**Setup.** Trained from the phase 7 checkpoint for up to 1.2M steps with a
`ConvergenceCallback` that probes every 50k steps and stops when plateau is
detected. Probes use `evaluate_model()` (deterministic, 100 episodes, seed 7)
to watch convergence without loading the model inside the seeded region (which
would shift episode sampling). Seed 7 is deliberately different from the final
eval seed (42) to avoid checkpoint selection on the reported episodes.

**Stopping rule.** Dual conditions, both checked on the last 4 probes:
- **Plateau:** goal rate span ≤ 4 percentage points AND intervention rate span ≤ 2 pp
- **No-improvement:** best goal rate not beaten for 6 consecutive probes

**Results:** Stopped at 700,000 steps (plateau condition triggered).

#### The training trajectory

| Probe | Step | Goal % | Intervention % | Lambda | Best? | Note |
|------|------|--------|----------------|--------|-------|------|
| 1 | 50k | 48.0 | 5.44 | 1.789 | — | Cold start under curriculum |
| 2 | 100k | 76.0 | 4.86 | 1.877 | — | Quick early climb |
| 3 | 150k | 64.0 | 5.98 | 1.940 | — | First oscillation down |
| 4 | 200k | 87.0 | 7.36 | 1.998 | Best (87%) | Early peak |
| 5 | 250k | 65.0 | 3.85 | 2.079 | — | Second major oscillation |
| 6 | 300k | 85.0 | 3.69 | 2.143 | — | Recovery |
| 7 | 350k | 85.0 | 3.71 | 2.212 | — | Stabilizing |
| 8 | 400k | 88.0 | 4.17 | 2.284 | Best (88%) | New peak |
| 9 | 450k | **100.0** | 6.13 | 2.345 | **Best** | **Peak before degradation** |
| 10 | 500k | 95.0 | 3.38 | 2.367 | — | Slight decline |
| 11 | 550k | 99.0 | 3.46 | 2.371 | — | Recovery, stays high |
| 12 | 600k | 97.0 | 3.96 | 2.384 | — | Stabilized |
| 13 | 650k | 98.0 | 4.34 | 2.415 | — | Plateau region |
| 14 | 700k | 97.0 | 2.55 | 2.416 | — | Final probe, stopping triggered |

**Pattern.** The policy oscillates rather than monotonically improves: 48% → 76%
→ 64% → 87% → 65% → 85% → 88% → 100% → 95% → 99% → 97% → 98% → 97%. The best
probe achieved 100% goal rate at step 450k (phase9_best checkpoint). Lambda
climbs monotonically (1.79 → 2.42), a signature of the constraint target (0.05
or 5% intervention rate) being unreachable under the current policy capacity —
dual ascent never stops pushing because the policy never satisfies the budget.

#### Probe baseline and proper comparison

In-training probes call `evaluate_model()`, which does NOT load the model inside
the seeded region. This means:
- A probe value is **NOT directly comparable to the authoritativeness number from
  `evaluate()` on the same checkpoint**, because the two functions sample from
  different episode sequences.
- What a probe value **IS** comparable to is the starting checkpoint probed at
  the same seed.

Baseline from phase 7 checkpoint (100 ep, seed 7, deterministic):

| Metric | Value | Purpose |
|--------|-------|---------|
| Goal rate | 79.0% | Like-for-like reference for reading run probes |
| Intervention rate | 6.41% | — |

Against this seed-7 baseline:
- The 300k probe (85.0% / 3.69%) beats the starting checkpoint on both metrics
- The 450k probe (100.0% / 6.13%) beats on goal rate but regresses slightly on intervention
- The final probes (97-99%) show stable, high performance at lower intervention than the peak

Best checkpoint was selected by **goal rate alone** (100% at 450k), but the open
question for phase 10 is whether a composite criterion (e.g., Pareto dominance
or goal×(1-intervention)) would better serve the project's values.

### Goal B: TensorBoard live-update metrics dashboard

Configured SB3's logger to stream scalar metrics to TensorBoard during training,
with custom callbacks to capture the dual-critic signals:

**Required scalars:**
- `rollout/episode_reward_mean` — task reward per episode
- `train/cost_value_estimate` — cost-value head's predicted cost over a batch
- `train/cost_return_observed` — empirical cumulative cost per episode
- `train/cost_value_bias` — the residual (estimate − observed)
- `train/lambda` — Lagrange multiplier at each update
- `rollout/intervention_rate_mean` — shield intervention rate
- `train/goal_arrival_rate_mean` — goal success rate in each batch
- `train/cost_critic_loss` — MSE on cost-value predictions

**Live dashboard during run:**
TensorBoard event file written to `saferl/eval/phase9/run_events/`. View with:
```bash
tensorboard --logdir saferl/eval/phase9/run_events/
```
Curves extend in real time as training progresses (sampled every 1–2 updates).

**Cost-critic breakdown.** Unlike a simple MSE-loss view:
- `cost_value_estimate` (model's prediction) = what the critic thinks will happen
- `cost_return_observed` (empirical trajectory cost) = what actually happened
- `cost_value_bias` (residual) = tells you if the cost signal is being systematized
  or still spiky. Declining bias over training confirms the dual-critic loss is
  converging, not just churning through bad episodes.

The breakdown is read from `train()` in `saferl/training/dual_critic.py` and
logged as three separate scalars (not just one combined MSE).

### Final held-out evaluation

**Checkpoint:** `saferl/eval/phase9/saferl_phase9_best.zip` (best goal rate from
probes, step 450k). MD5: `716fae1409d3eb3ae03bfa63b6663e65`.

**Protocol:** 500 episodes, deterministic, seed 42 (the reserved held-out seed).

**Results:**

| Metric | Value | vs. Phase 7 |
|--------|-------|------------|
| Goal rate | **90.8%** (454/500) | +1.6 pp |
| Intervention rate | 6.79% | **-5.78 pp** |
| Collisions | 0 | — |

The held-out eval confirms a modest goal-rate gain (90.8% vs 89.2%) with a
substantial intervention reduction (6.79% vs 12.57%). The policy under phase-9
long training is significantly more efficient at steering — 48% fewer
interventions per episode for 1.6pp more goals.

**Caveat on best-checkpoint selection.** The checkpoint was chosen by the best
probe value (100% goal at 450k). The held-out eval at the same checkpoint reads
90.8%, a 9.2pp drop from the probe. This 9pp spread is not noise:
- Probes run on 100 episodes at seed 7 (the selection seed)
- Held-out eval runs on 500 episodes at seed 42 (reserved)
- A 100-episode probe on identical weights has ~3pp std error; 9pp is real
  difference in generalization

This is why separate seeds matter: the probe did its job (identifying which
checkpoint is best *for that seed*) but the true capability on unseen episodes
was slightly lower. For phase 10: consider whether best-checkpoint selection
should account for this held-out variance — e.g., by probing at both seeds, or
by running mini-evals on the test seed periodically.

### Convergence finding: policy oscillation + unreachable target

The monotonically climbing lambda (1.79 → 2.42) despite plateau-detected probes
(goal span 2pp, intervention span 1.8pp) confirms the constraint target is not
being met. The policy oscillates around ~97% goal rate and ~4-6% intervention
rate, a local equilibrium under dual ascent. To push the intervention rate toward
the 5% target would require either:

1. **Longer training** (lambda was still rising at 700k — dual ascent had not
   converged)
2. **Lower constraint target** (5% may be unrealistic given the shield/policy
   co-evolution dynamics)
3. **Reward shaping for navigation efficiency** (still the most-deferred open
   question — the policy gets no credit for dodging hazards cleanly)

The oscillation pattern suggests the policy is sensitive to constraint tuning:
as lambda rises, intervention pressure tightens, the policy explores more
conservative trajectories, overfits locally, then finds a high-goal window before
lambda keeps climbing. This is not a bug (the dual-gradient updater is working)
but it is a sign that the single-critic constrained PPO has reached its
representational ceiling under this curriculum. A curriculum that starts tighter
or a multi-scale exploration approach might dampen the oscillation.

### Tests and code quality

All **40 tests pass** (unchanged from phase 8). Added:

- `saferl/training/train_phase9.py` — training script with convergence callbacks
- `saferl/eval/phase9/probe_baseline.json` — documented probe protocol and
  like-for-like baselines
- `saferl/eval/phase9/convergence_probes.csv` — all 14 probes, timestamps,
  best-checkpoint tracking
- `saferl/eval/phase9/final_eval/` — authoritative eval output (CSV, summary)
- TensorBoard event logs in `run_events/` (gitignored; 500MB+)

### What Phase 10 should assume

1. **The phase 9 checkpoint is 90.8% goal / 6.79% intervention** (500 ep,
   deterministic, seed 42, limited sensing). This is the authoritative figure
   for any phase-10 baseline or comparison.
2. **Policy oscillates under dual ascent.** Lambda reaches 2.42 (still rising
   at 700k) while the intervention rate plateaus around 4–6%. The 5% target
   appears unreachable under current architecture. Longer training, tighter
   constraint curriculum, or reward shaping are the levers to try.
3. **Best-checkpoint selection has held-out variance.** The 100-episode probe
   saw 100% goal; the 500-episode held-out eval saw 90.8%. A 9pp gap is real.
   Phase 10 should either probe at multiple seeds or run mini-evals on the
   reserved seed periodically during training.
4. **TensorBoard scalars now include cost-critic internals** (estimate,
   observed, bias). Use these to debug whether the cost signal is stable or
   spiky — don't rely on loss alone.
5. **Probes are a useful convergence monitor but not held-out evaluations.**
   They sample from a fixed seed during training (necessary for convergence
   detection), so checkpoint selection should be treated with the variance
   caveat above. The held-out eval protocol (`evaluate()` with seed 42) is
   separate.
