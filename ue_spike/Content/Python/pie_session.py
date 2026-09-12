"""PIE session driver for the SafeRL UE visualization.

Launch (full GUI editor -- NOT headless, NOT -nullrhi, NOT a commandlet):
    SAFERL_RUN_PIE=1 UnrealEditor SafeRLUESpike.uproject -log

`init_unreal.py` (auto-run by UE's Python plugin at startup) calls arm()
when SAFERL_RUN_PIE=1 is set.

Do NOT launch this with -ExecutePythonScript: that flag makes the editor
quit as soon as the script returns.

State machine (all driven from a slate post-tick callback):
    boot -> import meshes, build scene, request PIE
    waiting_for_pie -> poll until the PIE world exists and holds our actor
    running -> one env step per engine tick, timed; screenshot partway
    done -> stop counting, leave the editor and PIE session up

Phase 8 visual fidelity changes:
    - Satellite: composite actor (body + solar panels + antenna boom + dish)
    - Debris: imported irregular rock meshes (perturbed icospheres)
    - Sky: large inverted dome with dark material
    - Lighting: single harsh directional light, minimal ambient
"""
import json
import math
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
# per-debris rotation (degrees) for visual variety
DEBRIS_ROTATIONS = [
    (0, 0, 0),
    (45, 30, 0),
    (0, 60, 20),
    (25, 0, 70),
    (10, 45, 55),
]
# per-debris non-uniform scale for further variety
DEBRIS_SCALES = [
    (1.8, 1.4, 2.0),
    (1.5, 2.2, 1.3),
    (2.0, 1.6, 1.8),
    (1.3, 1.9, 2.1),
    (2.2, 1.5, 1.7),
]

SATELLITE_TAG = "SafeRLSatellite"
SATELLITE_PART_TAG = "SafeRLSatPart"
DEBRIS_TAG = "SafeRLDebris"
GOAL_TAG = "SafeRLGoal"
CAPTURE_TAG = "SafeRLCapture"
SKY_TAG = "SafeRLSky"
LIGHT_TAG = "SafeRLLight"
ALL_TAGS = (SATELLITE_TAG, SATELLITE_PART_TAG, DEBRIS_TAG, GOAL_TAG,
            CAPTURE_TAG, SKY_TAG, LIGHT_TAG)

NUM_STEPS = 500
BOOT_TICKS = 120
DT = 1.0 / 30.0
FORCE_ACCEL = 400.0
MAX_SPEED = 900.0
GOAL_RADIUS = 1.0 * SCALE

CAM_LOCATION = unreal.Vector(2600.0, -900.0, 1800.0)
CAM_ROTATION = unreal.Rotator(0.0, -34.5, 133.4)
ACTION_TABLE = {
    0: (0.0, FORCE_ACCEL),
    1: (0.0, -FORCE_ACCEL),
    2: (-FORCE_ACCEL, 0.0),
    3: (FORCE_ACCEL, 0.0),
}

_HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.normpath(os.path.join(_HERE, "..", "Meshes"))

UNCAP_FRAMERATE = os.environ.get("SAFERL_UNCAP") == "1"
STEPS_PER_TICK = int(os.environ.get("SAFERL_STEPS_PER_TICK", "1"))
RESULTS_NAME = os.environ.get("SAFERL_RESULTS_NAME", "pie_session_results.json")
RESULTS_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", RESULTS_NAME))

SPHERE = "/Engine/BasicShapes/Sphere.Sphere"
CUBE = "/Engine/BasicShapes/Cube.Cube"
CYLINDER = "/Engine/BasicShapes/Cylinder.Cylinder"
CONE = "/Engine/BasicShapes/Cone.Cone"

HEARTBEAT_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "pie_heartbeat.log")
)

# content path where imported debris meshes land
DEBRIS_CONTENT_PATH = "/Game/Meshes"


def _log(msg):
    unreal.log(f"[saferl-pie] {msg}")
    try:
        with open(HEARTBEAT_PATH, "a") as f:
            f.write(f"{time.time():.3f} {msg}\n")
    except Exception:
        pass


def _disable_background_throttle():
    try:
        settings = unreal.get_default_object(unreal.EditorPerformanceSettings)
        settings.set_editor_property("throttle_cpu_when_not_foreground", False)
        _log("disabled throttle_cpu_when_not_foreground")
    except Exception as e:
        _log(f"could not disable background throttle: {e!r}")


def _uncap_framerate(game_world):
    for cmd in ("t.MaxFPS 0", "r.VSync 0", "Slate.AllowThrottling 0"):
        try:
            unreal.SystemLibrary.execute_console_command(game_world, cmd)
            _log(f"applied CVar: {cmd}")
        except Exception as e:
            _log(f"CVar {cmd} failed (non-fatal): {e!r}")


def _env_to_world(env_pos):
    return unreal.Vector(env_pos[0] * SCALE, env_pos[1] * SCALE, env_pos[2] * SCALE)


def _subsys():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


# ── mesh import ─────────────────────────────────────────────────────────────

def _import_obj_meshes():
    """Import OBJ debris rock meshes into UE content if not already present."""
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    tasks = []

    for i in range(5):
        asset_name = f"debris_rock_{i}"
        content_path = f"{DEBRIS_CONTENT_PATH}/{asset_name}"
        if unreal.EditorAssetLibrary.does_asset_exist(content_path):
            _log(f"asset {content_path} already exists, skipping import")
            continue

        obj_path = os.path.join(MESH_DIR, f"{asset_name}.obj")
        if not os.path.isfile(obj_path):
            _log(f"WARNING: {obj_path} not found, debris {i} will use fallback cube")
            continue

        task = unreal.AssetImportTask()
        task.filename = obj_path
        task.destination_path = DEBRIS_CONTENT_PATH
        task.destination_name = asset_name
        task.automated = True
        task.save = True
        task.replace_existing = True
        tasks.append(task)
        _log(f"queued import: {obj_path} -> {content_path}")

    # sky dome
    sky_content = f"{DEBRIS_CONTENT_PATH}/sky_dome"
    if not unreal.EditorAssetLibrary.does_asset_exist(sky_content):
        sky_obj = os.path.join(MESH_DIR, "sky_dome.obj")
        if os.path.isfile(sky_obj):
            task = unreal.AssetImportTask()
            task.filename = sky_obj
            task.destination_path = DEBRIS_CONTENT_PATH
            task.destination_name = "sky_dome"
            task.automated = True
            task.save = True
            task.replace_existing = True
            tasks.append(task)
            _log(f"queued import: {sky_obj} -> {sky_content}")

    if tasks:
        asset_tools.import_asset_tasks(tasks)
        _log(f"imported {len(tasks)} mesh assets")
    else:
        _log("all mesh assets already imported")


def _load_debris_mesh(index):
    """Load an imported debris rock mesh, falling back to a basic shape."""
    asset_path = f"{DEBRIS_CONTENT_PATH}/debris_rock_{index}.debris_rock_{index}"
    mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
    if mesh is not None:
        return mesh
    # fallback: try without doubled name (import naming varies)
    asset_path = f"{DEBRIS_CONTENT_PATH}/debris_rock_{index}"
    mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
    if mesh is not None:
        return mesh
    _log(f"WARNING: could not load debris_rock_{index}, using fallback cone")
    return unreal.EditorAssetLibrary.load_asset(CONE)


def _load_sky_dome_mesh():
    """Load the imported sky dome mesh."""
    for path in (f"{DEBRIS_CONTENT_PATH}/sky_dome.sky_dome",
                 f"{DEBRIS_CONTENT_PATH}/sky_dome"):
        mesh = unreal.EditorAssetLibrary.load_asset(path)
        if mesh is not None:
            return mesh
    _log("WARNING: sky dome mesh not available, using fallback sphere")
    return unreal.EditorAssetLibrary.load_asset(SPHERE)


# ── satellite (composite actor from basic shapes) ───────────────────────────

def _spawn_satellite(env_pos):
    """Spawn a composite satellite: body + two solar panels + antenna boom."""
    subsys = _subsys()
    world_pos = _env_to_world(env_pos)

    # body: rectangular box
    body = subsys.spawn_actor_from_class(unreal.StaticMeshActor, world_pos)
    body.set_actor_label("SafeRL_Satellite_Body")
    body.tags = [SATELLITE_TAG]
    comp = body.static_mesh_component
    comp.set_static_mesh(unreal.EditorAssetLibrary.load_asset(CUBE))
    comp.set_mobility(unreal.ComponentMobility.MOVABLE)
    body.set_actor_scale3d(unreal.Vector(1.2, 0.8, 0.6))

    def attach_part(mesh_path, label, offset, scale, rotation=None):
        loc = unreal.Vector(world_pos.x + offset[0],
                            world_pos.y + offset[1],
                            world_pos.z + offset[2])
        part = subsys.spawn_actor_from_class(unreal.StaticMeshActor, loc)
        part.set_actor_label(label)
        part.tags = [SATELLITE_PART_TAG]
        c = part.static_mesh_component
        c.set_static_mesh(unreal.EditorAssetLibrary.load_asset(mesh_path))
        c.set_mobility(unreal.ComponentMobility.MOVABLE)
        part.set_actor_scale3d(unreal.Vector(*scale))
        if rotation:
            part.set_actor_rotation(unreal.Rotator(*rotation), False)
        part.attach_to_actor(body, "", unreal.AttachmentRule.KEEP_WORLD,
                             unreal.AttachmentRule.KEEP_WORLD,
                             unreal.AttachmentRule.KEEP_WORLD, True)
        return part

    # solar panel left (thin wide rectangle)
    attach_part(CUBE, "SafeRL_Satellite_PanelL",
                (-220, 0, 0), (0.05, 1.6, 0.5))

    # solar panel right
    attach_part(CUBE, "SafeRL_Satellite_PanelR",
                (220, 0, 0), (0.05, 1.6, 0.5))

    # antenna boom (thin cylinder rising from body)
    attach_part(CYLINDER, "SafeRL_Satellite_Antenna",
                (0, 0, 100), (0.1, 0.1, 1.2))

    # dish at top of antenna
    attach_part(SPHERE, "SafeRL_Satellite_Dish",
                (0, 0, 180), (0.3, 0.3, 0.15))

    _log(f"spawned composite satellite at env {env_pos}")
    return body


# ── scene construction ──────────────────────────────────────────────────────

def build_scene():
    """Build the full visual scene: satellite, debris, sky dome, lighting."""
    subsys = _subsys()

    # clear actors from a previous run
    for actor in subsys.get_all_level_actors():
        tags = [str(t) for t in (actor.tags or [])]
        if any(t in ALL_TAGS for t in tags):
            subsys.destroy_actor(actor)

    # remove default sky atmosphere / sky light / fog if present
    for actor in subsys.get_all_level_actors():
        cls_name = actor.get_class().get_name()
        if cls_name in ("SkyAtmosphere", "SkyLight", "ExponentialHeightFog",
                        "VolumetricCloud", "BP_Sky_Sphere_C"):
            subsys.destroy_actor(actor)
            _log(f"removed default {cls_name}")

    # import OBJ meshes if needed
    _import_obj_meshes()

    # ── sky dome (large dark sphere with inverted normals) ──
    sky_mesh = _load_sky_dome_mesh()
    sky = subsys.spawn_actor_from_class(
        unreal.StaticMeshActor,
        unreal.Vector(ENV_SIZE * SCALE / 2, ENV_SIZE * SCALE / 2, 0),
    )
    sky.set_actor_label("SafeRL_SkyDome")
    sky.tags = [SKY_TAG]
    sky_comp = sky.static_mesh_component
    sky_comp.set_static_mesh(sky_mesh)
    sky_comp.set_mobility(unreal.ComponentMobility.STATIC)
    _log("spawned sky dome")

    # ── satellite ──
    _spawn_satellite(START_ENV_POS)

    # ── goal marker (small bright sphere) ──
    goal = subsys.spawn_actor_from_class(
        unreal.StaticMeshActor, _env_to_world(GOAL_ENV_POS))
    goal.set_actor_label("SafeRL_Goal")
    goal.tags = [GOAL_TAG]
    goal_comp = goal.static_mesh_component
    goal_comp.set_static_mesh(unreal.EditorAssetLibrary.load_asset(SPHERE))
    goal_comp.set_mobility(unreal.ComponentMobility.MOVABLE)
    goal.set_actor_scale3d(unreal.Vector(1.5, 1.5, 1.5))
    _log("spawned goal marker")

    # ── debris field (imported rock meshes with varied scale/rotation) ──
    for i, pos in enumerate(DEBRIS_ENV_POS):
        debris_mesh = _load_debris_mesh(i)
        d = subsys.spawn_actor_from_class(
            unreal.StaticMeshActor, _env_to_world(pos))
        d.set_actor_label(f"SafeRL_Debris_{i}")
        d.tags = [DEBRIS_TAG]
        dc = d.static_mesh_component
        dc.set_static_mesh(debris_mesh)
        dc.set_mobility(unreal.ComponentMobility.MOVABLE)
        sx, sy, sz = DEBRIS_SCALES[i]
        d.set_actor_scale3d(unreal.Vector(sx, sy, sz))
        rx, ry, rz = DEBRIS_ROTATIONS[i]
        d.set_actor_rotation(unreal.Rotator(rx, ry, rz), False)
        _log(f"spawned debris {i} at env {pos}")

    # ── lighting: single harsh directional light (distant sun) ──
    sun = subsys.spawn_actor_from_class(
        unreal.DirectionalLight,
        unreal.Vector(0, 0, 1500),
    )
    sun.set_actor_label("SafeRL_Sun")
    sun.tags = [LIGHT_TAG]
    # pitch down at ~30 degrees, rotated to cast shadows across the field
    sun.set_actor_rotation(unreal.Rotator(0.0, -30.0, 160.0), False)
    light_comp = sun.light_component
    light_comp.set_editor_property("intensity", 8.0)
    # harsh shadows, no volumetric scattering
    light_comp.set_editor_property("cast_shadows", True)
    try:
        light_comp.set_editor_property("atmospheric_sun_light", False)
    except Exception:
        pass
    try:
        light_comp.set_editor_property("use_temperature", True)
        light_comp.set_editor_property("temperature", 5800.0)
    except Exception:
        pass
    _log("spawned directional sun light")

    # ── scene capture for screenshots ──
    cap = subsys.spawn_actor_from_class(
        unreal.SceneCapture2D, CAM_LOCATION, CAM_ROTATION,
    )
    cap.set_actor_label("SafeRL_Capture")
    cap.tags = [CAPTURE_TAG]
    cap.capture_component2d.set_editor_property("capture_every_frame", False)
    cap.capture_component2d.set_editor_property("capture_on_movement", False)

    unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(
        CAM_LOCATION, CAM_ROTATION,
    )
    _log("scene built (phase 8 visual fidelity)")


def capture_png(game_world, file_name):
    """Render the live PIE scene through the SceneCapture2D and write a PNG."""
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
        comp.texture_target = None
        _log(f"wrote {os.path.join(out_dir, file_name)}")
        return True
    except Exception as e:
        import traceback
        _log(f"screenshot failed (non-fatal): {e!r}")
        unreal.log_error(traceback.format_exc())
        return False


# ── PIE bridge: drives the satellite actor during the step loop ─────────────

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
    "run_start_tick": 0,
}


def _find_satellite_in_pie():
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
        "engine_ticks_used": _state["ticks"] - _state["run_start_tick"],
        "steps_per_tick": STEPS_PER_TICK,
        "framerate_uncapped": UNCAP_FRAMERATE,
        "mode": "live PIE session, full GUI editor",
    }
    _log(f"DONE {result}")
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    _log(f"wrote {RESULTS_PATH}")
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
            if UNCAP_FRAMERATE:
                _uncap_framerate(game_world)
            _log(f"protocol: {NUM_STEPS} steps, steps_per_tick={STEPS_PER_TICK}, "
                 f"uncapped={UNCAP_FRAMERATE}")
            _state["run_start_tick"] = _state["ticks"]
            _state["t0"] = time.perf_counter()
            _state["phase"] = "running"
            return

        if _state["phase"] == "posing":
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
        for _ in range(STEPS_PER_TICK):
            if _state["count"] >= NUM_STEPS:
                break
            obs = bridge.step(bridge.action_toward_goal())
            _state["count"] += 1
            if bridge.at_goal():
                _state["arrivals"] += 1
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
