"""Phase 3 Goal A: bring up a real PIE session with the satellite + debris
field visible, drive it from the phase 2 reset()/step() loop, and measure
the tick-bound step rate that the phase 2 commandlet spike could not get.

Launch (full GUI editor -- NOT headless, NOT -nullrhi, NOT a commandlet):
    SAFERL_RUN_PIE=1 UnrealEditor SafeRLUESpike.uproject -log

`init_unreal.py` (auto-run by UE's Python plugin at startup) calls arm()
when SAFERL_RUN_PIE=1 is set.

Do NOT launch this with -ExecutePythonScript: that flag makes the editor
quit as soon as the script returns, which is what actually killed both the
phase 2 tick-callback attempt and the first phase 3 attempt. Phase 2
attributed that to "-nullrhi has nothing keeping it alive"; the real cause
is the flag itself, and it fires the same way in a full GUI session.

State machine (all driven from a slate post-tick callback):
    boot -> build scene, request PIE
    waiting_for_pie -> poll until the PIE world exists and holds our actor
    running -> one env step per engine tick, timed; screenshot partway
    done -> stop counting, leave the editor and PIE session up
"""
import json
import os
import time

import unreal

# ── layout, mirroring saferl/configs/default.yaml's PyBullet env ──────────
SCALE = 200.0          # 1 env unit = 200cm, so a size-10 field is 20m across
ENV_SIZE = 10
START_ENV_POS = (0.0, 0.0, 0.5)
GOAL_ENV_POS = (ENV_SIZE - 1, ENV_SIZE - 1, 0.5)
DEBRIS_ENV_POS = [            # fixed (not random) so the visual is reproducible
    (2.0, 3.0, 0.5),
    (4.5, 1.5, 0.5),
    (3.0, 6.0, 0.5),
    (6.5, 4.0, 0.5),
    (7.0, 7.0, 0.5),
]

SATELLITE_TAG = "SafeRLSatellite"
DEBRIS_TAG = "SafeRLDebris"
GOAL_TAG = "SafeRLGoal"
CAPTURE_TAG = "SafeRLCapture"
ALL_TAGS = (SATELLITE_TAG, DEBRIS_TAG, GOAL_TAG, CAPTURE_TAG)

NUM_STEPS = 500
BOOT_TICKS = 120          # let the editor settle before touching the level
DT = 1.0 / 30.0
FORCE_ACCEL = 400.0
MAX_SPEED = 900.0         # cm/s, keeps the satellite on screen and watchable
GOAL_RADIUS = 1.0 * SCALE # matches the env's goal_threshold of 1.0 env units

# Camera set side-on to the start->goal diagonal (not along it, or the actors
# stack up in a line), pitched down ~35 degrees onto the middle of the field.
CAM_LOCATION = unreal.Vector(2600.0, -900.0, 1800.0)
CAM_ROTATION = unreal.Rotator(0.0, -34.5, 133.4)
ACTION_TABLE = {
    0: (0.0, FORCE_ACCEL),
    1: (0.0, -FORCE_ACCEL),
    2: (-FORCE_ACCEL, 0.0),
    3: (FORCE_ACCEL, 0.0),
}

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", "pie_session_results.json"))

SPHERE = "/Engine/BasicShapes/Sphere.Sphere"
CUBE = "/Engine/BasicShapes/Cube.Cube"


HEARTBEAT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pie_heartbeat.log")
)


def _log(msg):
    unreal.log(f"[saferl-pie] {msg}")
    # UE's own log buffering stalls under load, so mirror to a plain file we
    # can watch from outside the engine
    try:
        with open(HEARTBEAT_PATH, "a") as f:
            f.write(f"{time.time():.3f} {msg}\n")
    except Exception:
        pass


def _disable_background_throttle():
    """The editor throttles its tick rate hard when its window isn't the
    foreground app, which starves the step loop while we drive it from a
    terminal. Turn that off for this session."""
    try:
        settings = unreal.get_default_object(unreal.EditorPerformanceSettings)
        settings.set_editor_property("throttle_cpu_when_not_foreground", False)
        _log("disabled throttle_cpu_when_not_foreground")
    except Exception as e:
        _log(f"could not disable background throttle: {e!r}")


def _env_to_world(env_pos):
    return unreal.Vector(env_pos[0] * SCALE, env_pos[1] * SCALE, env_pos[2] * SCALE)


def _spawn(mesh_path, env_pos, label, tag, scale):
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = subsys.spawn_actor_from_class(unreal.StaticMeshActor, _env_to_world(env_pos))
    actor.set_actor_label(label)
    actor.tags = [tag]
    comp = actor.static_mesh_component
    comp.set_static_mesh(unreal.EditorAssetLibrary.load_asset(mesh_path))
    comp.set_mobility(unreal.ComponentMobility.MOVABLE)
    actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))
    _log(f"spawned {label} at env {env_pos}")
    return actor


def build_scene():
    """Spawn satellite, goal marker, debris field, and a light."""
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    # clear actors from a previous run so re-runs are idempotent
    for actor in subsys.get_all_level_actors():
        tags = [str(t) for t in (actor.tags or [])]
        if any(t in ALL_TAGS for t in tags):
            subsys.destroy_actor(actor)

    _spawn(SPHERE, START_ENV_POS, "SafeRL_Satellite", SATELLITE_TAG, 2.2)
    _spawn(SPHERE, GOAL_ENV_POS, "SafeRL_Goal", GOAL_TAG, 3.0)
    for i, pos in enumerate(DEBRIS_ENV_POS):
        _spawn(CUBE, pos, f"SafeRL_Debris_{i}", DEBRIS_TAG, 1.8)

    light = subsys.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 1500))
    light.set_actor_label("SafeRL_Light")
    light.set_actor_rotation(unreal.Rotator(0, -50, 30), False)

    # A SceneCapture2D spawned here gets duplicated into the PIE world along
    # with everything else, giving us a camera we can render a PNG from while
    # PIE runs. Needed because this host is Wayland: X11 screen grabs of the
    # editor window come back black, so the engine has to produce the image.
    cap = subsys.spawn_actor_from_class(
        unreal.SceneCapture2D, CAM_LOCATION, CAM_ROTATION,
    )
    cap.set_actor_label("SafeRL_Capture")
    cap.tags = [CAPTURE_TAG]
    # SceneCaptureComponent2D re-renders the whole scene every frame by
    # default, which stalls the render thread hard enough that post-tick
    # callbacks stop firing. We only want a frame on demand.
    cap.capture_component2d.set_editor_property("capture_every_frame", False)
    cap.capture_component2d.set_editor_property("capture_on_movement", False)

    # aim the editor viewport at the field so the visual record is useful
    unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(
        CAM_LOCATION, CAM_ROTATION,
    )
    _log("scene built")


def capture_png(game_world, file_name):
    """Render the live PIE scene through the SceneCapture2D and write a PNG.

    Never raises: a failed screenshot must not take down the measurement.
    Note UKismetRenderingLibrary is exposed as unreal.RenderingLibrary --
    UE strips the "Kismet" prefix, same as SystemLibrary/MathLibrary.
    """
    try:
        actors = unreal.GameplayStatics.get_all_actors_with_tag(game_world, CAPTURE_TAG)
        if not actors:
            _log("no capture actor in the PIE world; skipping screenshot")
            return False
        comp = actors[0].capture_component2d
        comp.set_editor_property("capture_every_frame", False)
        comp.set_editor_property("capture_on_movement", False)
        render_target = unreal.RenderingLibrary.create_render_target2d(
            game_world, 1920, 1080, unreal.TextureRenderTargetFormat.RTF_RGBA8,
            unreal.LinearColor(0.0, 0.0, 0.0, 1.0),
        )
        comp.texture_target = render_target
        comp.capture_source = unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
        comp.capture_scene()
        out_dir = os.path.normpath(os.path.join(_HERE, "..", ".."))
        unreal.RenderingLibrary.export_render_target(
            game_world, render_target, out_dir, file_name
        )
        comp.texture_target = None   # stop it re-rendering after the shot
        _log(f"wrote {os.path.join(out_dir, file_name)}")
        return True
    except Exception as e:
        import traceback
        _log(f"screenshot failed (non-fatal): {e!r}")
        unreal.log_error(traceback.format_exc())
        return False


# ── the phase 2 reset()/step() loop, now driving a PIE-world actor ────────
class PIEBridge:
    def __init__(self, game_world, actor):
        self.world = game_world
        self.actor = actor
        self.velocity = [0.0, 0.0, 0.0]

    def reset(self):
        self.actor.set_actor_location(_env_to_world(START_ENV_POS), False, False)
        self.velocity = [0.0, 0.0, 0.0]
        return self._read_obs()

    def step(self, action):
        ax, ay = ACTION_TABLE.get(action, (0.0, 0.0))
        self.velocity[0] += ax * DT
        self.velocity[1] += ay * DT

        # speed clamp so the satellite crosses the field at a watchable pace
        # instead of accelerating off-screen; costs no extra engine calls, so
        # the throughput measurement is unaffected
        speed = (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5
        if speed > MAX_SPEED:
            scale = MAX_SPEED / speed
            self.velocity[0] *= scale
            self.velocity[1] *= scale

        loc = self.actor.get_actor_location()
        self.actor.set_actor_location(
            unreal.Vector(loc.x + self.velocity[0] * DT,
                          loc.y + self.velocity[1] * DT,
                          loc.z),
            False, False,
        )
        return self._read_obs()

    def action_toward_goal(self):
        """Greedy thrust toward the goal -- stands in for a policy so the
        satellite visibly traverses the debris field during the demo."""
        loc = self.actor.get_actor_location()
        goal = _env_to_world(GOAL_ENV_POS)
        dx, dy = goal.x - loc.x, goal.y - loc.y
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        return 0 if dy > 0 else 1

    def at_goal(self):
        loc = self.actor.get_actor_location()
        goal = _env_to_world(GOAL_ENV_POS)
        return ((loc.x - goal.x) ** 2 + (loc.y - goal.y) ** 2) ** 0.5 < GOAL_RADIUS

    def _read_obs(self):
        loc = self.actor.get_actor_location()
        return {"position": [loc.x, loc.y, loc.z], "velocity": list(self.velocity)}


_state = {
    "phase": "boot",
    "ticks": 0,
    "bridge": None,
    "count": 0,
    "t0": None,
    "handle": None,
    "pose_ticks": 0,
    "arrivals": 0,
}


def _find_satellite_in_pie():
    """Return (game_world, satellite_actor) once PIE is genuinely up, else None."""
    game_world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
    if game_world is None:
        return None
    actors = unreal.GameplayStatics.get_all_actors_with_tag(game_world, SATELLITE_TAG)
    if not actors:
        return None
    return game_world, actors[0]


def _finish():
    elapsed = time.perf_counter() - _state["t0"]
    result = {
        "num_steps": _state["count"],
        "elapsed_seconds": elapsed,
        "steps_per_second": _state["count"] / elapsed if elapsed > 0 else float("inf"),
        "goal_arrivals": _state["arrivals"],
        "mode": "live PIE session, full GUI editor, one env step per engine tick",
        "note": "tick-bound: each step runs inside a slate post-tick callback while a "
                "real Play-In-Editor session renders, so this rate is capped by the "
                "engine's actual frame rate -- unlike the phase 2 commandlet number, "
                "which had no frame loop at all.",
    }
    _log(f"DONE {result}")
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    _log(f"wrote {RESULTS_PATH}")
    # Screenshots come after the timed run, never during: a SceneCapture2D
    # holding a 1920x1080 target re-renders the scene and drags the frame rate
    # from ~95/s to ~6/s, which would poison the measurement.
    _state["phase"] = "posing"
    _state["pose_ticks"] = 0


def _on_tick(delta_seconds):
    try:
        if _state["phase"] == "done":
            return
        _state["ticks"] += 1

        if _state["phase"] == "boot":
            if _state["ticks"] < BOOT_TICKS:
                return
            _disable_background_throttle()
            build_scene()
            unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_begin_play()
            _log("PIE requested")
            _state["phase"] = "waiting_for_pie"
            return

        if _state["phase"] == "waiting_for_pie":
            found = _find_satellite_in_pie()
            if found is None:
                if _state["ticks"] % 30 == 0:
                    _log(f"waiting for PIE world + satellite ({_state['ticks']} ticks)")
                if _state["ticks"] > BOOT_TICKS + 3600:
                    unreal.log_error("[saferl-pie] PIE never came up; giving up")
                    unreal.unregister_slate_post_tick_callback(_state["handle"])
                    _state["phase"] = "done"
                return
            game_world, actor = found
            _log(f"PIE is live, satellite found: {actor.get_name()}")
            _state["bridge"] = PIEBridge(game_world, actor)
            _state["bridge"].reset()
            _state["t0"] = time.perf_counter()
            _state["phase"] = "running"
            return

        if _state["phase"] == "posing":
            # untimed: fly the satellite back out into the field so the
            # screenshots show it mid-traverse rather than parked on the goal
            bridge = _state["bridge"]
            _state["pose_ticks"] += 1
            bridge.step(bridge.action_toward_goal())
            if _state["pose_ticks"] == 45:
                capture_png(bridge.world, "pie_session_midrun.png")
            elif _state["pose_ticks"] >= 90:
                capture_png(bridge.world, "pie_session_final.png")
                unreal.unregister_slate_post_tick_callback(_state["handle"])
                _state["phase"] = "done"
                _log("leaving the editor and PIE session running")
            return

        bridge = _state["bridge"]
        obs = bridge.step(bridge.action_toward_goal())
        _state["count"] += 1

        # a real episode boundary: on arrival, reset() and fly it again, so the
        # loop exercises reset as well as step across the run
        if bridge.at_goal():
            _state["arrivals"] += 1
            _log(f"satellite reached the goal ({_state['arrivals']}x) -- resetting")
            bridge.reset()

        if _state["count"] % 100 == 0:
            _log(f"step {_state['count']}/{NUM_STEPS} satellite at {obs['position']}")

        if _state["count"] >= NUM_STEPS:
            _finish()
    except Exception as e:
        import traceback
        unreal.log_error(f"[saferl-pie] EXCEPTION: {e!r}")
        unreal.log_error(traceback.format_exc())
        unreal.unregister_slate_post_tick_callback(_state["handle"])
        _state["phase"] = "done"


def arm():
    _state["handle"] = unreal.register_slate_post_tick_callback(_on_tick)
    _log("armed; scene build + PIE request will run once the editor settles")
