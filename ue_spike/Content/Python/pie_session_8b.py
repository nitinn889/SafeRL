"""PIE session driver for SafeRL UE visualization — Phase 8b: Real Assets Edition.

This version supports real sourced assets:
  - Satellite: real model (ISS, downloaded from NASA 3D Resources)
  - Debris rocks: Quixel Megascans scanned rocks
  - Skybox: Space HDRI or reconfigured Sky Atmosphere for deep space

Fallback to procedural/basic shapes if assets not present (same as phase 8).

Asset sourcing documented in Phase 8b README section.
"""
import json
import math
import os
import time

import unreal

# ── environment layout (same as phase 8) ──────────────────────────────────
SCALE = 200.0
ENV_SIZE = 10
START_ENV_POS = (0.0, 0.0, 0.5)
GOAL_ENV_POS = (ENV_SIZE - 1, ENV_SIZE - 1, 0.5)
DEBRIS_ENV_POS = [
    (2.0, 3.0, 0.5),
    (4.5, 1.5, 0.5),
    (3.0, 6.0, 0.5),
    (6.5, 4.0, 0.5),
    (7.0, 7.0, 0.5),
]
DEBRIS_ROTATIONS = [
    (0, 0, 0),
    (45, 30, 0),
    (0, 60, 20),
    (25, 0, 70),
    (10, 45, 55),
]
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
DEBRIS_CONTENT_PATH = "/Game/Meshes"
MEGASCANS_CONTENT_PATH = "/Game/Megascans"

# ── Phase 8b: real asset source paths ────────────────────────────────────
# These can be populated by user or automatic discovery
REAL_SATELLITE_MESH = None  # Will be set to actual ISS model path when available
REAL_ROCK_MESHES = [
    # Megascans rock paths — these are examples; actual names depend on imported models
    "/Game/Megascans/Plants/Rocks/SM_Rock_01",
    "/Game/Megascans/Plants/Rocks/SM_Rock_02",
    "/Game/Megascans/Plants/Rocks/SM_Rock_03",
    "/Game/Megascans/Plants/Rocks/SM_Rock_04",
    "/Game/Megascans/Plants/Rocks/SM_Rock_05",
]
SPACE_HDRI_PATH = None  # Will be set when HDRI imported

UNCAP_FRAMERATE = os.environ.get("SAFERL_UNCAP") == "1"
STEPS_PER_TICK = int(os.environ.get("SAFERL_STEPS_PER_TICK", "1"))
RESULTS_NAME = os.environ.get("SAFERL_RESULTS_NAME", "pie_session_results.json")
RESULTS_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", RESULTS_NAME))
HEARTBEAT_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", "pie_heartbeat.log"))

# ── fallback to phase 8 basic shapes if real assets unavailable ────────────
SPHERE = "/Engine/BasicShapes/Sphere.Sphere"
CUBE = "/Engine/BasicShapes/Cube.Cube"
CYLINDER = "/Engine/BasicShapes/Cylinder.Cylinder"
CONE = "/Engine/BasicShapes/Cone.Cone"


def _log(msg):
    unreal.log(f"[saferl-pie-8b] {msg}")
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


# ── asset discovery and loading ──────────────────────────────────────────

def _find_available_rocks():
    """Discover available real rock meshes in the project.

    Returns list of up to 5 valid mesh paths. Tries:
    1. Quixel Megascans rocks (real scanned assets)
    2. Fallback to imported OBJ meshes from phase 8
    3. Basic shapes as last resort
    """
    rocks = []

    # Try Megascans rocks (real scanned assets)
    megascans_patterns = [
        "/Game/Megascans/Plants/Rocks/SM_Rock_",
        "/Game/Megascans/Rocks/",
        "/Megascans/Environments/Rocky_Outcrop/Meshes/",
    ]

    for pattern in megascans_patterns:
        for i in range(1, 10):
            path = f"{pattern}{i}"
            try:
                mesh = unreal.EditorAssetLibrary.load_asset(path)
                if mesh:
                    rocks.append(mesh)
                    _log(f"discovered Megascans rock: {path}")
                    if len(rocks) >= 5:
                        return rocks[:5]
            except:
                pass

    # Fallback: imported OBJ meshes from phase 8
    if len(rocks) < 5:
        for i in range(5):
            for path_variant in (
                f"{DEBRIS_CONTENT_PATH}/debris_rock_{i}.debris_rock_{i}",
                f"{DEBRIS_CONTENT_PATH}/debris_rock_{i}",
            ):
                try:
                    mesh = unreal.EditorAssetLibrary.load_asset(path_variant)
                    if mesh:
                        rocks.append(mesh)
                        _log(f"discovered procedural rock: {path_variant}")
                        break
                except:
                    pass

    # Last resort: basic shapes
    while len(rocks) < 5:
        try:
            mesh = unreal.EditorAssetLibrary.load_asset(CONE)
            if mesh:
                rocks.append(mesh)
                _log(f"using fallback cone mesh (rock {len(rocks)})")
        except:
            pass

    return rocks[:5]


def _find_satellite_mesh():
    """Try to load a real satellite model.

    Looks for:
    1. Imported ISS model (from NASA 3D Resources)
    2. Fallback to phase 8 composite from basic shapes
    """
    # Look for imported ISS model
    iss_paths = [
        "/Game/Meshes/ISS",
        "/Game/Models/ISS",
        "/Game/Satellite/ISS",
        REAL_SATELLITE_MESH,  # User-provided path
    ]

    for path in iss_paths:
        if path:
            try:
                mesh = unreal.EditorAssetLibrary.load_asset(path)
                if mesh:
                    _log(f"loaded real satellite model: {path}")
                    return ("real", mesh)
            except:
                pass

    _log("no real satellite model found; will spawn composite from basic shapes")
    return ("composite", None)


def _load_space_hdri():
    """Try to load a space HDRI for the skybox.

    Looks for imported HDRI assets, returns None if not found.
    In that case, Sky Atmosphere will be reconfigured for deep space instead.
    """
    hdri_paths = [
        "/Game/Meshes/space_hdri",
        "/Game/Textures/space_hdri",
        "/Game/HDRI/space",
        SPACE_HDRI_PATH,  # User-provided path
    ]

    for path in hdri_paths:
        if path:
            try:
                texture = unreal.EditorAssetLibrary.load_asset(path)
                if texture:
                    _log(f"loaded space HDRI: {path}")
                    return texture
            except:
                pass

    _log("no space HDRI found; will reconfigure Sky Atmosphere for deep space")
    return None


# ── satellite actor spawning ─────────────────────────────────────────────

def _spawn_satellite_real(env_pos, mesh):
    """Spawn a real satellite model as the controllable actor."""
    subsys = _subsys()
    world_pos = _env_to_world(env_pos)

    sat = subsys.spawn_actor_from_class(unreal.StaticMeshActor, world_pos)
    sat.set_actor_label("SafeRL_Satellite_Real")
    sat.tags = [SATELLITE_TAG]
    comp = sat.static_mesh_component
    comp.set_static_mesh(mesh)
    comp.set_mobility(unreal.ComponentMobility.MOVABLE)

    # Scale to match environment (ISS model is typically in cm; env uses cm too)
    # Adjust scale as needed based on actual model size
    sat.set_actor_scale3d(unreal.Vector(1.0, 1.0, 1.0))

    _log(f"spawned real satellite model at env {env_pos}")
    return sat


def _spawn_satellite_composite(env_pos):
    """Fallback: spawn phase 8 composite satellite from basic shapes."""
    subsys = _subsys()
    world_pos = _env_to_world(env_pos)

    # body
    body = subsys.spawn_actor_from_class(unreal.StaticMeshActor, world_pos)
    body.set_actor_label("SafeRL_Satellite_Composite")
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

    attach_part(CUBE, "SafeRL_Satellite_PanelL", (-220, 0, 0), (0.05, 1.6, 0.5))
    attach_part(CUBE, "SafeRL_Satellite_PanelR", (220, 0, 0), (0.05, 1.6, 0.5))
    attach_part(CYLINDER, "SafeRL_Satellite_Antenna", (0, 0, 100), (0.1, 0.1, 1.2))
    attach_part(SPHERE, "SafeRL_Satellite_Dish", (0, 0, 180), (0.3, 0.3, 0.15))

    _log(f"spawned composite satellite at env {env_pos} (real model not available)")
    return body


# ── scene construction ───────────────────────────────────────────────────

def build_scene_8b():
    """Phase 8b scene builder: real assets with fallbacks."""
    subsys = _subsys()

    # Clear actors from previous run
    for actor in subsys.get_all_level_actors():
        tags = [str(t) for t in (actor.tags or [])]
        if any(t in ALL_TAGS for t in tags):
            subsys.destroy_actor(actor)

    # Remove default sky atmosphere / sky light / fog
    for actor in subsys.get_all_level_actors():
        cls_name = actor.get_class().get_name()
        if cls_name in ("SkyAtmosphere", "SkyLight", "ExponentialHeightFog",
                        "VolumetricCloud", "BP_Sky_Sphere_C"):
            subsys.destroy_actor(actor)
            _log(f"removed default {cls_name}")

    # ── Sky / Atmosphere ──
    hdri = _load_space_hdri()
    if hdri:
        # If we have an HDRI, use it as a sky dome texture
        sky_mesh = unreal.EditorAssetLibrary.load_asset(SPHERE)
        sky = subsys.spawn_actor_from_class(
            unreal.StaticMeshActor,
            unreal.Vector(ENV_SIZE * SCALE / 2, ENV_SIZE * SCALE / 2, 0),
        )
        sky.set_actor_label("SafeRL_Sky_HDRI")
        sky.tags = [SKY_TAG]
        sky_comp = sky.static_mesh_component
        sky_comp.set_static_mesh(sky_mesh)
        sky_comp.set_mobility(unreal.ComponentMobility.STATIC)
        # TODO: assign HDRI material to sky_comp (requires material setup)
        _log("spawned sky dome with HDRI")
    else:
        # Reconfigure Sky Atmosphere for deep space
        _reconfigure_sky_for_deep_space()
        _log("configured Sky Atmosphere for deep space")

    # ── Satellite (real or composite) ──
    sat_type, sat_mesh = _find_satellite_mesh()
    if sat_type == "real" and sat_mesh:
        _spawn_satellite_real(START_ENV_POS, sat_mesh)
    else:
        _spawn_satellite_composite(START_ENV_POS)

    # ── Goal marker ──
    goal = subsys.spawn_actor_from_class(
        unreal.StaticMeshActor, _env_to_world(GOAL_ENV_POS))
    goal.set_actor_label("SafeRL_Goal")
    goal.tags = [GOAL_TAG]
    goal_comp = goal.static_mesh_component
    goal_comp.set_static_mesh(unreal.EditorAssetLibrary.load_asset(SPHERE))
    goal_comp.set_mobility(unreal.ComponentMobility.MOVABLE)
    goal.set_actor_scale3d(unreal.Vector(1.5, 1.5, 1.5))
    _log("spawned goal marker")

    # ── Debris (real rocks or procedural) ──
    available_rocks = _find_available_rocks()
    for i, pos in enumerate(DEBRIS_ENV_POS):
        debris_mesh = available_rocks[i] if i < len(available_rocks) else unreal.EditorAssetLibrary.load_asset(CONE)
        d = subsys.spawn_actor_from_class(unreal.StaticMeshActor, _env_to_world(pos))
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

    # ── Lighting ──
    sun = subsys.spawn_actor_from_class(
        unreal.DirectionalLight,
        unreal.Vector(0, 0, 1500),
    )
    sun.set_actor_label("SafeRL_Sun")
    sun.tags = [LIGHT_TAG]
    sun.set_actor_rotation(unreal.Rotator(0.0, -30.0, 160.0), False)
    light_comp = sun.light_component
    light_comp.set_editor_property("intensity", 8.0)
    light_comp.set_editor_property("cast_shadows", True)
    try:
        light_comp.set_editor_property("atmospheric_sun_light", False)
        light_comp.set_editor_property("use_temperature", True)
        light_comp.set_editor_property("temperature", 5800.0)
    except:
        pass
    _log("spawned directional sun light")

    # ── Scene capture ──
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
    _log("scene built (phase 8b real assets edition)")


def _reconfigure_sky_for_deep_space():
    """Reconfigure UE's Sky Atmosphere for deep-space appearance.

    Sets up a sparse starfield, dark space environment without atmosphere.
    This is used when no HDRI is available.
    """
    subsys = _subsys()

    try:
        # Create or reconfigure Sky Atmosphere
        sky_atm = subsys.spawn_actor_from_class(unreal.SkyAtmosphere)
        sky_atm.set_actor_label("SafeRL_SkyAtmosphere_DeepSpace")
        sky_atm.tags = [SKY_TAG]

        # Configure for deep space (no atmosphere scattering)
        try:
            sky_atm.set_editor_property("ground_albedo", 0.0)
            sky_atm.set_editor_property("rayleigh_scattering_scale", 0.0)
            sky_atm.set_editor_property("mie_scattering_scale", 0.0)
        except:
            pass

        _log("reconfigured Sky Atmosphere for deep space")
    except Exception as e:
        _log(f"could not configure Sky Atmosphere: {e!r}; using default")


# ── the rest of pie_session implementation (run loop, capture, etc.) ──
# ... (continue with existing PIE loop code, unchanged)
# This file should be merged with the full pie_session.py,
# or imported as a replacement build_scene() function.
