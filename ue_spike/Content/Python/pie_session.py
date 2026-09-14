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

Phase 8c real-asset changes (replaces phase 8's procedural geometry):
    - Satellite: NASA's Advanced Composition Explorer (ACE) model, downloaded
      from science.nasa.gov/3d-resources (public domain), imported from GLB.
      ISS was searched for first and returned zero results in NASA's 3D
      Resources catalog; ACE is a real, public-domain NASA satellite and the
      "next reasonable option" per the phase 8c brief.
    - Debris: 5 "moon_rock_01".."moon_rock_05" models from Poly Haven
      (CC0, real photogrammetry scans), imported from glTF+bin+textures.
    - Sky: NASA SVS "Deep Star Maps 2020" (public domain), a real all-sky
      map built from 1.7 billion stars in the Hipparcos-2/Tycho-2/Gaia DR2
      catalogs, imported as an EXR texture and applied via an unlit emissive
      material on the existing inverted-dome geometry (the dome shape itself
      is a standard skybox technique; what changed is a real astronomical
      texture replacing the flat placeholder material).
    - All real assets have a composite/procedural fallback if import fails,
      logged explicitly rather than failing silently.

3D space mode (SAFERL_DIMS=3, set by run_ue_demo.sh --config space3d.yaml):
    - Builds in its own empty level (/Game/Maps/SpaceLevel, generated on first
      run) instead of the default Open World template, whose Landscape was the
      checkerboard floor. Black void, one hard sun, locked manual exposure.
    - Mirrors SAFERL_NUM_DEBRIS rocks (cycling the scanned meshes) moving on all
      three axes, with the goal at the far corner of the cube.
    - Wide and chase cameras (SAFERL_CAMERA), switchable live by writing
      "wide" or "chase" to ue_spike/camera_mode. Satellite heading and rock
      tumble are visual only; the env simulates neither.
"""
import json
import math
import os
import random
import time

import unreal

# ── layout, mirroring the PyBullet env the launcher's config describes ──────
# UE's embedded Python 3.11 can't be assumed to have PyYAML, so run_ue_demo.sh
# reads the config with the project venv and passes the few values the scene
# needs as environment variables. Defaults are the planar default.yaml.
SCALE = 200.0          # 1 env unit = 200cm, so a size-10 field is 20m across
DIMS = int(os.environ.get("SAFERL_DIMS", "2"))
ENV_SIZE = int(os.environ.get("SAFERL_ENV_SIZE", "10"))
NUM_DEBRIS = int(os.environ.get("SAFERL_NUM_DEBRIS", "5"))
SPACE_LEVEL = os.environ.get("SAFERL_SPACE_LEVEL", "1" if DIMS == 3 else "0") == "1"
SPACE_LEVEL_PATH = "/Game/Maps/SpaceLevel"
if DIMS == 3:
    START_ENV_POS = (0.0, 0.0, 0.0)
    GOAL_ENV_POS = (ENV_SIZE - 1, ENV_SIZE - 1, ENV_SIZE - 1)
else:
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


def _debris_spawn_pos(i):
    """Where debris i sits before the live bridge starts moving it.

    The first five planar slots keep phase 8's hand-placed positions, so the
    2D scene is unchanged. Any further slot, and every 3D slot, gets a
    deterministic position inside the field derived from its index.
    """
    if DIMS == 2 and i < len(DEBRIS_ENV_POS):
        return DEBRIS_ENV_POS[i]
    r = random.Random(1000 + i)
    lo, hi = 1.0, ENV_SIZE - 2.0
    x, y = r.uniform(lo, hi), r.uniform(lo, hi)
    return (x, y, r.uniform(lo, hi) if DIMS == 3 else 0.5)


SATELLITE_TAG = "SafeRLSatellite"
SATELLITE_PART_TAG = "SafeRLSatPart"
DEBRIS_TAG = "SafeRLDebris"
GOAL_TAG = "SafeRLGoal"
CAPTURE_TAG = "SafeRLCapture"
SKY_TAG = "SafeRLSky"
STAR_TAG = "SafeRLStar"
LIGHT_TAG = "SafeRLLight"
CAMERA_TAG = "SafeRLCamera"
POSTFX_TAG = "SafeRLPostFX"
PLAYER_START_TAG = "SafeRLPlayerStart"
ALL_TAGS = (SATELLITE_TAG, SATELLITE_PART_TAG, DEBRIS_TAG, GOAL_TAG,
            CAPTURE_TAG, SKY_TAG, STAR_TAG, LIGHT_TAG, CAMERA_TAG, POSTFX_TAG,
            PLAYER_START_TAG)

NUM_STEPS = 500
BOOT_TICKS = 120
DT = 1.0 / 30.0
FORCE_ACCEL = 400.0
MAX_SPEED = 900.0
GOAL_RADIUS = 1.0 * SCALE

# Camera. The defaults are phase 3's framing, kept because they show the
# debris field and the agent's path clearly -- that is what the demo is of.
# A consequence, diagnosed in phase 10: at pitch -34.5 the ground plane
# fills the frame edge to edge and the horizon sits above the top of the
# image, so *no sky is in shot at all*. That, not material brightness, is
# why the real NASA starmap never showed up in phase 8c's captures. The
# overrides below exist to take an establishing shot that does include sky,
# without disturbing the framing the demo itself uses.
CAM_LOCATION = unreal.Vector(
    float(os.environ.get("SAFERL_CAM_X", "2600.0")),
    float(os.environ.get("SAFERL_CAM_Y", "-900.0")),
    float(os.environ.get("SAFERL_CAM_Z", "1800.0")),
)
CAM_ROTATION = unreal.Rotator(
    0.0,
    float(os.environ.get("SAFERL_CAM_PITCH", "-34.5")),
    float(os.environ.get("SAFERL_CAM_YAW", "133.4")),
)
# Camera mode. "fixed" is the framing above (phase 3's, and the 2D default).
# "wide" frames the whole field from outside a corner; "chase" follows the
# satellite from behind and above along its smoothed velocity. Switch while
# running by writing "wide"/"chase"/"fixed" to CAMERA_MODE_PATH.
CAMERA_MODE = os.environ.get("SAFERL_CAMERA", "wide" if DIMS == 3 else "fixed")
CHASE_BACK_CM = 900.0
CHASE_UP_CM = 350.0
CHASE_LERP = 0.12          # fraction of the remaining gap closed per tick
# Space look. Manual exposure, so auto-exposure can't adapt to whatever fills
# the frame and crush the stars or blow out the rocks. Defaults are tuned by
# measuring captures; all three are overridable.
EXPOSURE_BIAS = float(os.environ.get("SAFERL_EXPOSURE_BIAS", "0.0"))
SUN_LUX = float(os.environ.get("SAFERL_SUN_LUX", "8.0"))
ACTION_TABLE = {
    0: (0.0, FORCE_ACCEL),
    1: (0.0, -FORCE_ACCEL),
    2: (-FORCE_ACCEL, 0.0),
    3: (FORCE_ACCEL, 0.0),
}

_HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.normpath(os.path.join(_HERE, "..", "Meshes"))
SATELLITE_SRC_DIR = os.path.join(MESH_DIR, "satellite")
ROCKS_SRC_DIR = os.path.join(MESH_DIR, "rocks")
SKYBOX_SRC_DIR = os.path.join(MESH_DIR, "skybox")

UNCAP_FRAMERATE = os.environ.get("SAFERL_UNCAP") == "1"
STEPS_PER_TICK = int(os.environ.get("SAFERL_STEPS_PER_TICK", "1"))
RESULTS_NAME = os.environ.get("SAFERL_RESULTS_NAME", "pie_session_results.json")

# Live-mirror mode: instead of running the scripted 500-step benchmark with
# the move-toward-goal heuristic, poll a state file published by
# saferl.demo.live_policy_bridge and move the UE actors to match a real
# trained-policy rollout running in a separate process. Separate process
# because UE 5.8 embeds Python 3.11 and this project's venv is 3.14 --
# torch/SB3 cannot be imported into UE's interpreter at all.
LIVE_POLICY = os.environ.get("SAFERL_LIVE_POLICY") == "1"
LIVE_STATE_NAME = os.environ.get("SAFERL_LIVE_STATE", "live_policy_state.json")
# Frames between live-mode screenshots; 0 disables. A SceneCapture2D
# re-render is expensive (phase 3 measured it dragging the loop from ~119 to
# ~6 steps/sec when left on every frame), so this is deliberately sparse.
LIVE_CAPTURE_EVERY = int(os.environ.get("SAFERL_LIVE_CAPTURE_EVERY", "450"))
# >0 switches capture to a numbered sequence of this many frames (plus a
# metrics sidecar), for assembling the demo animation. 0 = alternating slots.
LIVE_CAPTURE_SEQ = int(os.environ.get("SAFERL_LIVE_CAPTURE_SEQ", "0"))
RESULTS_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", RESULTS_NAME))
LIVE_STATE_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", LIVE_STATE_NAME))
CAMERA_MODE_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", "camera_mode"))

SPHERE = "/Engine/BasicShapes/Sphere.Sphere"
CUBE = "/Engine/BasicShapes/Cube.Cube"
CYLINDER = "/Engine/BasicShapes/Cylinder.Cylinder"
CONE = "/Engine/BasicShapes/Cone.Cone"

# ── phase 8c real-asset config ──────────────────────────────────────────────
SATELLITE_CONTENT_PATH = "/Game/Meshes/Satellite"
ROCKS_CONTENT_PATH = "/Game/Meshes/Rocks"
TEXTURES_CONTENT_PATH = "/Game/Textures"

ACE_GLB_NAME = "ACE_satellite"
# moon_rock_01 excluded: its Poly Haven glTF package ships 4 separate LOD
# mesh nodes (LOD0..LOD3) instead of the single-mesh structure the other
# rocks use, which Interchange import handled unreliably; moon_rock_06
# substituted in its place (see README phase 8c for the investigation).
# moon_rock_02 excluded: quantitatively the roundest of the scanned rocks
# (vertex-radius coefficient of variation 0.108, vs 0.20-0.30 for the
# others -- measured directly from the raw glTF vertex data, not guessed).
# It IS genuinely real scanned geometry (confirmed: correct asset path,
# correct non-uniform vertex count, correct material, verified even with
# Nanite forced off), but at this scene's single-harsh-light setup and
# render distance it was visually indistinguishable from a plain sphere in
# the screenshot, which defeats the point of using a real asset. Swapped
# for moon_rock_07 (CV 0.30, unambiguously irregular).
MOON_ROCK_NAMES = ["moon_rock_03", "moon_rock_04", "moon_rock_05",
                    "moon_rock_06", "moon_rock_07"]
STARMAP_EXR_NAME = "nasa_starmap_2020_4k"
# Emissive gain on the starmap; see _get_or_create_starmap_material.
STARMAP_EMISSIVE_GAIN = float(os.environ.get("SAFERL_STARMAP_GAIN", "60.0"))
# The goal is a beacon, not a lit object: under the dim space sun a default
# grey sphere rendered near-black. Unlit emissive keeps it readable at any
# exposure; this scales its brightness.
GOAL_EMISSIVE_GAIN = float(os.environ.get("SAFERL_GOAL_GAIN", "3.0"))
EMISSIVE_BASE_MATERIAL = "M_Emissive_Unlit"

# Point stars (3D space level only). Phase 11 measured the NASA dome texture
# reading as a dim smear rather than stars even at 33x emissive gain, so stars
# are drawn as geometry instead: tiny unlit emissive spheres on a shell inside
# the dome. These are PROCEDURAL, not catalogue positions -- seeded, uniform
# on the sphere with a denser tilted band standing in for the galactic plane,
# and most stars faint, as real star counts are. The NASA dome stays behind
# them as the diffuse glow.
NUM_STARS = int(os.environ.get("SAFERL_NUM_STARS", "1500"))
STAR_SHELL_CM = 40000.0
# (share of stars, diameter cm, emissive gain). At 400 m and ~21 px/deg on a
# 1920 px frame, 60 cm is ~2 px and 140 cm ~4 px.
STAR_TIERS = ((0.70, 60.0, 1.5), (0.25, 90.0, 5.0), (0.05, 140.0, 20.0))
STAR_TINTS = ((0.75, 0.85, 1.0), (1.0, 1.0, 1.0), (1.0, 0.9, 0.7), (1.0, 0.75, 0.5))
# The gain is a material parameter, set on a material instance each launch, so
# it can be tuned without rebuilding the material. "_v2" because the first
# version baked the gain in as a constant.
STARMAP_BASE_MATERIAL = "M_Starmap_Sky_v2"
STARMAP_INSTANCE = "MI_Starmap_Sky"

SATELLITE_TARGET_MAX_DIM_CM = 500.0   # longest dimension after rescale
ROCK_TARGET_MAX_DIM_CM = [190, 230, 160, 210, 250]  # per-rock variety

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


# ── mesh import (phase 8: procedural OBJ fallback) ──────────────────────────

def _import_obj_meshes():
    """Import OBJ debris rock meshes into UE content if not already present.

    This is the phase 8 procedural fallback path (perturbed icospheres +
    inverted-dome sky), used only if phase 8c's real-asset import fails.
    """
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


def _load_debris_mesh_procedural(index):
    """Load a phase 8 procedural debris rock mesh, falling back to a cone."""
    asset_path = f"{DEBRIS_CONTENT_PATH}/debris_rock_{index}.debris_rock_{index}"
    mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
    if mesh is not None:
        return mesh
    asset_path = f"{DEBRIS_CONTENT_PATH}/debris_rock_{index}"
    mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
    if mesh is not None:
        return mesh
    _log(f"WARNING: could not load debris_rock_{index}, using fallback cone")
    return unreal.EditorAssetLibrary.load_asset(CONE)


def _load_sky_dome_mesh_procedural():
    """Load the phase 8 procedural inverted sky dome mesh."""
    for path in (f"{DEBRIS_CONTENT_PATH}/sky_dome.sky_dome",
                 f"{DEBRIS_CONTENT_PATH}/sky_dome"):
        mesh = unreal.EditorAssetLibrary.load_asset(path)
        if mesh is not None:
            return mesh
    _log("WARNING: sky dome mesh not available, using fallback sphere")
    return unreal.EditorAssetLibrary.load_asset(SPHERE)


# ── mesh import (phase 8c: real sourced assets) ──────────────────────────────

def _disable_nanite(mesh, content_path):
    """Force off Nanite's auto-generated coarse proxy on an imported mesh.

    Found by direct visual inspection: one of the imported rocks (a fairly
    round scan) rendered in the SceneCapture2D screenshot as a smooth grey
    sphere with a visible UV-sphere seam, even though Python confirmed the
    correct 914-vertex mesh, correct asset path, and correct material were
    all assigned to the component (get_num_vertices/get_material both
    checked out). A neighbouring, more irregular rock rendered its full
    detail correctly. The distinguishing factor is shape: Nanite is enabled
    by default on Interchange glTF import, and its auto-generated fallback
    proxy -- used by some render paths including scene capture -- collapsed
    the roundest rock down to something that reads as a bare sphere while
    leaving jagged rocks visibly jagged even after simplification. Disabling
    Nanite forces the actual imported geometry to render everywhere.
    """
    try:
        settings = mesh.get_editor_property("nanite_settings")
        settings.set_editor_property("enabled", False)
        mesh.set_editor_property("nanite_settings", settings)
        unreal.EditorAssetLibrary.save_asset(content_path)
        _log(f"disabled Nanite on {content_path}")
    except Exception as e:
        _log(f"WARNING: could not disable Nanite on {content_path} "
             f"(non-fatal, mesh may render via Nanite proxy): {e!r}")


def _import_real_satellite():
    """Import the ACE satellite GLB (NASA 3D Resources, public domain).

    Returns the imported UStaticMesh, or None if the source file is missing
    or import fails (caller falls back to the phase 8 composite).
    """
    content_path = f"{SATELLITE_CONTENT_PATH}/{ACE_GLB_NAME}"
    existing = unreal.EditorAssetLibrary.load_asset(content_path)
    if existing is not None:
        _log(f"real satellite already imported: {content_path}")
        _disable_nanite(existing, content_path)
        return existing

    glb_path = os.path.join(SATELLITE_SRC_DIR, "ACE_satellite.glb")
    if not os.path.isfile(glb_path):
        _log(f"WARNING: real satellite source not found at {glb_path}")
        return None

    task = unreal.AssetImportTask()
    task.filename = glb_path
    task.destination_path = SATELLITE_CONTENT_PATH
    task.destination_name = ACE_GLB_NAME
    task.automated = True
    task.save = True
    task.replace_existing = True
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    try:
        asset_tools.import_asset_tasks([task])
    except Exception as e:
        _log(f"WARNING: satellite GLB import raised {e!r}")
        return None

    imported = task.get_objects()
    mesh = None
    for obj in imported:
        if isinstance(obj, unreal.StaticMesh):
            mesh = obj
            break
    if mesh is None:
        # fall back to a direct asset-registry load in case get_objects()
        # returned something else (e.g. an actor or a different asset type)
        mesh = unreal.EditorAssetLibrary.load_asset(content_path)
    if mesh is not None:
        _log(f"imported real satellite mesh: {content_path}")
        _disable_nanite(mesh, content_path)
    else:
        _log(f"WARNING: satellite GLB import produced no StaticMesh at {content_path}")
    return mesh


def _import_real_rocks():
    """Import the 5 Poly Haven moon_rock glTF models (CC0).

    Returns a list of up to 5 UStaticMesh objects (shorter than 5 if some
    imports failed -- caller pads with procedural/cone fallbacks).
    """
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    meshes = []
    for name in MOON_ROCK_NAMES:
        content_path = f"{ROCKS_CONTENT_PATH}/{name}"
        existing = unreal.EditorAssetLibrary.load_asset(content_path)
        if existing is not None:
            _log(f"real rock already imported: {content_path}")
            _disable_nanite(existing, content_path)
            meshes.append(existing)
            continue

        gltf_path = os.path.join(ROCKS_SRC_DIR, name, f"{name}.gltf")
        if not os.path.isfile(gltf_path):
            _log(f"WARNING: real rock source not found at {gltf_path}")
            continue

        task = unreal.AssetImportTask()
        task.filename = gltf_path
        task.destination_path = ROCKS_CONTENT_PATH
        task.destination_name = name
        task.automated = True
        task.save = True
        task.replace_existing = True
        try:
            asset_tools.import_asset_tasks([task])
        except Exception as e:
            _log(f"WARNING: rock {name} glTF import raised {e!r}")
            continue

        mesh = None
        for obj in task.get_objects():
            if isinstance(obj, unreal.StaticMesh):
                mesh = obj
                break
        if mesh is None:
            mesh = unreal.EditorAssetLibrary.load_asset(content_path)
        if mesh is not None:
            _log(f"imported real rock mesh: {content_path}")
            _disable_nanite(mesh, content_path)
            meshes.append(mesh)
        else:
            _log(f"WARNING: rock {name} glTF import produced no StaticMesh")

    return meshes


def _import_starmap_texture():
    """Import the NASA SVS Deep Star Maps 2020 EXR as a UTexture2D.

    Returns the texture, or None if the source file is missing or import
    fails (caller falls back to the phase 8 flat dark material).
    """
    content_path = f"{TEXTURES_CONTENT_PATH}/{STARMAP_EXR_NAME}"
    existing = unreal.EditorAssetLibrary.load_asset(content_path)
    if existing is not None:
        _log(f"starmap texture already imported: {content_path}")
        return existing

    exr_path = os.path.join(SKYBOX_SRC_DIR, "nasa_starmap_2020_4k.exr")
    if not os.path.isfile(exr_path):
        _log(f"WARNING: starmap EXR source not found at {exr_path}")
        return None

    task = unreal.AssetImportTask()
    task.filename = exr_path
    task.destination_path = TEXTURES_CONTENT_PATH
    task.destination_name = STARMAP_EXR_NAME
    task.automated = True
    task.save = True
    task.replace_existing = True
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    try:
        asset_tools.import_asset_tasks([task])
    except Exception as e:
        _log(f"WARNING: starmap EXR import raised {e!r}")
        return None

    texture = None
    for obj in task.get_objects():
        if isinstance(obj, unreal.Texture):
            texture = obj
            break
    if texture is None:
        texture = unreal.EditorAssetLibrary.load_asset(content_path)
    if texture is not None:
        _log(f"imported starmap texture: {content_path}")
    else:
        _log("WARNING: starmap EXR import produced no Texture asset")
    return texture


def _ensure_nanite_usage(material, content_path):
    """Make sure the starmap material carries the Nanite usage flag, saved.

    The sky dome mesh is imported from OBJ with UE's default Nanite setting,
    and a material applied to a Nanite mesh needs bUsedWithNanite. Without
    it the editor patches the flag in memory at load and logs a MapCheck
    warning; a cooked build cannot patch it and falls back to the default
    material. This material is generated here and its .uasset is gitignored,
    so the flag has to be set by this script -- clicking the editor's "Fix"
    would only repair the local copy until the next fresh build.
    """
    try:
        if material.get_editor_property("used_with_nanite"):
            _log(f"starmap material already has Nanite usage flag: {content_path}")
            return
        material.set_editor_property("used_with_nanite", True)
        unreal.MaterialEditingLibrary.recompile_material(material)
        unreal.EditorAssetLibrary.save_asset(content_path)
        _log(f"set + saved Nanite usage flag on {content_path}")
    except Exception as e:
        _log(f"WARNING: could not set Nanite usage flag on {content_path}: {e!r}")


def _get_or_create_starmap_material(texture):
    """Build (or reuse) an unlit emissive starmap material, and return an
    instance of it with the current gain applied.

    Unlit emissive shows the texture regardless of scene lighting -- the
    standard technique for a textured skybox dome. The texture is scaled by a
    "Gain" parameter before it reaches emissive: the NASA map is linear HDR
    data whose stars are, faithfully, very dim.
    """
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    base_path = f"{TEXTURES_CONTENT_PATH}/{STARMAP_BASE_MATERIAL}"
    base = unreal.EditorAssetLibrary.load_asset(base_path)
    if base is None:
        base = asset_tools.create_asset(
            STARMAP_BASE_MATERIAL, TEXTURES_CONTENT_PATH, unreal.Material,
            unreal.MaterialFactoryNew())
        mel = unreal.MaterialEditingLibrary
        tex_expr = mel.create_material_expression(
            base, unreal.MaterialExpressionTextureSample, -350, 0)
        tex_expr.texture = texture
        try:
            base.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
        except Exception as e:
            _log(f"could not set unlit shading model (non-fatal): {e!r}")
        gain = mel.create_material_expression(
            base, unreal.MaterialExpressionScalarParameter, -350, 250)
        gain.set_editor_property("parameter_name", "Gain")
        gain.set_editor_property("default_value", STARMAP_EMISSIVE_GAIN)
        mult = mel.create_material_expression(
            base, unreal.MaterialExpressionMultiply, -170, 0)
        mel.connect_material_expressions(tex_expr, "RGB", mult, "A")
        mel.connect_material_expressions(gain, "", mult, "B")
        mel.connect_material_property(mult, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        try:
            base.set_editor_property("used_with_nanite", True)
        except Exception as e:
            _log(f"WARNING: could not set Nanite usage flag on new starmap material: {e!r}")
        mel.recompile_material(base)
        unreal.EditorAssetLibrary.save_asset(base_path)
        _log(f"built unlit starmap material: {base_path}")
    else:
        _ensure_nanite_usage(base, base_path)

    inst_path = f"{TEXTURES_CONTENT_PATH}/{STARMAP_INSTANCE}"
    inst = unreal.EditorAssetLibrary.load_asset(inst_path)
    if inst is None:
        inst = asset_tools.create_asset(
            STARMAP_INSTANCE, TEXTURES_CONTENT_PATH, unreal.MaterialInstanceConstant,
            unreal.MaterialInstanceConstantFactoryNew())
        unreal.MaterialEditingLibrary.set_material_instance_parent(inst, base)
    unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(
        inst, "Gain", STARMAP_EMISSIVE_GAIN)
    unreal.MaterialEditingLibrary.update_material_instance(inst)
    unreal.EditorAssetLibrary.save_asset(inst_path)
    _log(f"starmap material instance gain={STARMAP_EMISSIVE_GAIN}")
    return inst


def _get_or_create_emissive_instance(inst_name, color, gain):
    """Unlit emissive Color x Gain. One shared base material; each use gets
    its own constant instance (editor Python has no
    MaterialInstanceDynamic.create). Unlit, so it reads at any exposure."""
    mel = unreal.MaterialEditingLibrary
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    base_path = f"{TEXTURES_CONTENT_PATH}/{EMISSIVE_BASE_MATERIAL}"
    base = unreal.EditorAssetLibrary.load_asset(base_path)
    if base is None:
        base = tools.create_asset(EMISSIVE_BASE_MATERIAL, TEXTURES_CONTENT_PATH,
                                  unreal.Material, unreal.MaterialFactoryNew())
        base.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
        col = mel.create_material_expression(
            base, unreal.MaterialExpressionVectorParameter, -350, 0)
        col.set_editor_property("parameter_name", "Color")
        col.set_editor_property("default_value", unreal.LinearColor(1.0, 1.0, 1.0, 1.0))
        g = mel.create_material_expression(
            base, unreal.MaterialExpressionScalarParameter, -350, 200)
        g.set_editor_property("parameter_name", "Gain")
        g.set_editor_property("default_value", 1.0)
        mult = mel.create_material_expression(
            base, unreal.MaterialExpressionMultiply, -170, 0)
        mel.connect_material_expressions(col, "", mult, "A")
        mel.connect_material_expressions(g, "", mult, "B")
        mel.connect_material_property(mult, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        mel.recompile_material(base)
        unreal.EditorAssetLibrary.save_asset(base_path)
        _log(f"built unlit emissive material: {base_path}")
    inst_path = f"{TEXTURES_CONTENT_PATH}/{inst_name}"
    inst = unreal.EditorAssetLibrary.load_asset(inst_path)
    if inst is None:
        inst = tools.create_asset(inst_name, TEXTURES_CONTENT_PATH,
                                  unreal.MaterialInstanceConstant,
                                  unreal.MaterialInstanceConstantFactoryNew())
        mel.set_material_instance_parent(inst, base)
    mel.set_material_instance_vector_parameter_value(
        inst, "Color", unreal.LinearColor(color[0], color[1], color[2], 1.0))
    mel.set_material_instance_scalar_parameter_value(inst, "Gain", gain)
    mel.update_material_instance(inst)
    unreal.EditorAssetLibrary.save_asset(inst_path)
    return inst


def _spawn_star_field(subsys, center):
    """NUM_STARS procedural point stars on a shell around `center`."""
    import math
    rng = random.Random(4242)
    sphere = unreal.EditorAssetLibrary.load_asset(SPHERE)
    mats = {(tier, t): _get_or_create_emissive_instance(f"MI_Star_{tier}_{t}", tint, gain)
            for tier, (_, _, gain) in enumerate(STAR_TIERS)
            for t, tint in enumerate(STAR_TINTS)}
    cum = [sum(x[0] for x in STAR_TIERS[:k + 1]) for k in range(len(STAR_TIERS))]
    tilt = math.radians(60.0)
    for _ in range(NUM_STARS):
        u = rng.random()
        tier = next(k for k, c in enumerate(cum) if u < c or k == len(cum) - 1)
        z = rng.uniform(-1.0, 1.0)
        if rng.random() < 0.35:          # the band
            z *= 0.12
        phi = rng.uniform(0.0, 2.0 * math.pi)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        x, y = r * math.cos(phi), r * math.sin(phi)
        y, z = y * math.cos(tilt) - z * math.sin(tilt), y * math.sin(tilt) + z * math.cos(tilt)
        loc = unreal.Vector(center.x + x * STAR_SHELL_CM, center.y + y * STAR_SHELL_CM,
                            center.z + z * STAR_SHELL_CM)
        star = subsys.spawn_actor_from_class(unreal.StaticMeshActor, loc)
        star.tags = [STAR_TAG]
        comp = star.static_mesh_component
        comp.set_static_mesh(sphere)
        comp.set_material(0, mats[(tier, rng.randrange(len(STAR_TINTS)))])
        comp.set_editor_property("cast_shadow", False)
        comp.set_mobility(unreal.ComponentMobility.STATIC)
        scale = STAR_TIERS[tier][1] / 100.0 * rng.uniform(0.85, 1.15)
        star.set_actor_scale3d(unreal.Vector(scale, scale, scale))
    _log(f"star field: {NUM_STARS} procedural point stars on a "
         f"{STAR_SHELL_CM / 100:.0f} m shell")


def _measure_and_rescale(actor, target_max_dim_cm):
    """Measure an actor's real bounding box (at scale 1,1,1) and apply a
    uniform scale so its longest dimension equals target_max_dim_cm.

    This replaces guessing a scale factor: real sourced meshes carry whatever
    units their authoring tool used, so the only reliable way to size them
    consistently is to measure the actual imported geometry and compute the
    scale from that measurement.
    """
    actor.set_actor_scale3d(unreal.Vector(1.0, 1.0, 1.0))
    origin, extent = actor.get_actor_bounds(False)
    size = (extent.x * 2.0, extent.y * 2.0, extent.z * 2.0)
    max_dim = max(size) if max(size) > 1e-6 else 1.0
    scale = target_max_dim_cm / max_dim
    actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))
    _log(f"measured {actor.get_actor_label()} raw size={tuple(round(s,1) for s in size)}cm "
         f"-> scale={scale:.4f} (target longest dim {target_max_dim_cm}cm)")
    return size, scale


# ── satellite: phase 8c real ACE mesh, phase 8 composite as fallback ───────

def _spawn_satellite_real(env_pos, mesh):
    """Spawn the real NASA ACE satellite mesh as the single controllable
    actor. Scale is computed by measuring the actual imported geometry
    (see _measure_and_rescale) rather than assumed."""
    subsys = _subsys()
    world_pos = _env_to_world(env_pos)

    sat = subsys.spawn_actor_from_class(unreal.StaticMeshActor, world_pos)
    sat.set_actor_label("SafeRL_Satellite_ACE")
    sat.tags = [SATELLITE_TAG]
    comp = sat.static_mesh_component
    comp.set_static_mesh(mesh)
    comp.set_mobility(unreal.ComponentMobility.MOVABLE)
    _measure_and_rescale(sat, SATELLITE_TARGET_MAX_DIM_CM)

    _log(f"spawned REAL satellite (NASA ACE model) at env {env_pos}")
    return sat


def _spawn_satellite_composite(env_pos):
    """Spawn a composite satellite: body + two solar panels + antenna boom.

    Phase 8 fallback, used only if the phase 8c real-asset import fails.
    """
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

    _log(f"spawned FALLBACK composite satellite at env {env_pos} "
         f"(real ACE mesh not available)")
    return body


# ── space level, camera poses, exposure ──────────────────────────────────────

def _ensure_space_level():
    """Switch the editor into the project's own empty level, creating it once.

    Without a level of its own the editor opens the default Open World
    template, whose Landscape is the checkerboard floor under every phase
    8-10 capture. new_level() creates, saves and loads a blank
    non-partitioned level; load_level() reopens it on later launches. The
    .umap is generated, so it is gitignored like the imported assets.
    """
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if unreal.EditorAssetLibrary.does_asset_exist(SPACE_LEVEL_PATH):
        ok = les.load_level(SPACE_LEVEL_PATH)
        _log(f"loaded space level {SPACE_LEVEL_PATH}: {ok}")
    else:
        ok = les.new_level(SPACE_LEVEL_PATH, False)
        _log(f"created space level {SPACE_LEVEL_PATH}: {ok}")
    return ok


def _camera_pose(mode):
    """(location, rotation) for the fixed and wide modes; chase starts wide."""
    if mode == "fixed":
        return CAM_LOCATION, CAM_ROTATION
    span = ENV_SIZE * SCALE
    c = span / 2.0
    if DIMS == 3:
        # Side-on to the start->goal diagonal, slightly above, far enough that
        # the diagonal's +-0.87 span fits inside the vertical half-FOV (~29 deg
        # at 90 deg horizontal, 16:9). The first pose, from above one corner,
        # left the start corner 36 deg off-axis and the satellite out of shot.
        d = 1.9 * span
        loc = unreal.Vector(c + 0.55 * d, c - 0.80 * d, c + 0.25 * d)
        target = unreal.Vector(c, c, c)
    else:
        loc = unreal.Vector(-0.55 * span, -0.35 * span, 0.9 * span)
        target = unreal.Vector(c, c, 0.0)
    return loc, unreal.MathLibrary.find_look_at_rotation(loc, target)


def _apply_manual_exposure(settings):
    """Lock exposure. Each property is set on its own, so a name that differs
    in this engine version is logged instead of silently skipping the rest."""
    wanted = (
        ("override_auto_exposure_method", True),
        ("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL),
        ("override_auto_exposure_apply_physical_camera_exposure", True),
        ("auto_exposure_apply_physical_camera_exposure", False),
        ("override_auto_exposure_bias", True),
        ("auto_exposure_bias", EXPOSURE_BIAS),
    )
    for name, value in wanted:
        try:
            settings.set_editor_property(name, value)
        except Exception as e:
            _log(f"WARNING: post-process property {name} not set: {e!r}")
    return settings


def _spawn_space_postfx(subsys, capture_actor):
    """An unbound manual-exposure post-process volume for the view, plus the
    same settings on the scene capture, which renders with its own copy."""
    vol = subsys.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0))
    vol.set_actor_label("SafeRL_SpacePostFX")
    vol.tags = [POSTFX_TAG]
    try:
        vol.set_editor_property("unbound", True)
        vol.set_editor_property(
            "settings", _apply_manual_exposure(vol.get_editor_property("settings")))
    except Exception as e:
        _log(f"WARNING: post-process volume setup failed: {e!r}")
    try:
        cc = capture_actor.capture_component2d
        cc.set_editor_property("post_process_blend_weight", 1.0)
        cc.set_editor_property(
            "post_process_settings",
            _apply_manual_exposure(cc.get_editor_property("post_process_settings")))
    except Exception as e:
        _log(f"WARNING: scene-capture exposure setup failed: {e!r}")
    _log(f"space post-process: manual exposure, bias {EXPOSURE_BIAS}")


# ── scene construction ──────────────────────────────────────────────────────

def build_scene():
    """Build the full visual scene: satellite, debris, sky dome, lighting.

    Phase 8c: tries real sourced assets first (NASA ACE satellite, Poly Haven
    moon rocks, NASA starmap skybox); each falls back independently to its
    phase 8 procedural equivalent if the source file is missing or import
    fails, with every fallback explicitly logged (never silent).
    """
    subsys = _subsys()
    asset_report = {"satellite": None, "rocks": None, "sky": None}

    # clear actors from a previous run
    for actor in subsys.get_all_level_actors():
        tags = [str(t) for t in (actor.tags or [])]
        if any(t in ALL_TAGS for t in tags):
            subsys.destroy_actor(actor)

    # remove default sky atmosphere / sky light / fog / sky sphere if
    # present. A plain default StaticMeshActor labeled "SM_SkySphere" (not
    # the "BP_Sky_Sphere_C" blueprint class this list used to check for)
    # survived every previous cleanup pass -- a template leftover this list
    # never matched. PlayerStart is deliberately left alone: removing it was
    # tried as a hypothesis for an unrelated rendering issue (see README) and
    # instead made PIE hang indefinitely with no PlayerStart to spawn from.
    for actor in subsys.get_all_level_actors():
        cls_name = actor.get_class().get_name()
        label = actor.get_actor_label()
        if (cls_name in ("SkyAtmosphere", "SkyLight", "ExponentialHeightFog",
                          "VolumetricCloud", "BP_Sky_Sphere_C")
                or label == "SM_SkySphere"):
            subsys.destroy_actor(actor)
            _log(f"removed default {cls_name} ({label})")

    if SPACE_LEVEL:
        # A new blank level has no PlayerStart, and PIE with nowhere to spawn
        # its default Pawn hangs (found in phase 10). One goes well outside the
        # field; the Pawn it spawns is culled once PIE is up.
        ps = subsys.spawn_actor_from_class(
            unreal.PlayerStart, unreal.Vector(-2000.0, -2000.0, 0.0))
        ps.set_actor_label("SafeRL_PlayerStart")
        ps.tags = [PLAYER_START_TAG]

    # import phase 8 procedural OBJ meshes (used as fallback source only)
    _import_obj_meshes()

    # ── sky dome: real NASA starmap texture on the inverted-dome geometry,
    #    falls back to phase 8's flat dark material if the EXR/material
    #    pipeline fails ──
    sky_mesh = _load_sky_dome_mesh_procedural()
    sky = subsys.spawn_actor_from_class(
        unreal.StaticMeshActor,
        unreal.Vector(ENV_SIZE * SCALE / 2, ENV_SIZE * SCALE / 2,
                      ENV_SIZE * SCALE / 2 if DIMS == 3 else 0),
    )
    sky.set_actor_label("SafeRL_SkyDome")
    sky.tags = [SKY_TAG]
    sky_comp = sky.static_mesh_component
    sky_comp.set_static_mesh(sky_mesh)
    sky_comp.set_mobility(unreal.ComponentMobility.STATIC)

    starmap_tex = _import_starmap_texture()
    if starmap_tex is not None:
        try:
            starmap_mat = _get_or_create_starmap_material(starmap_tex)
            sky_comp.set_material(0, starmap_mat)
            asset_report["sky"] = "real (NASA Deep Star Maps 2020)"
            _log("spawned sky dome with REAL NASA starmap material")
        except Exception as e:
            asset_report["sky"] = f"FALLBACK (material build failed: {e!r})"
            _log(f"WARNING: starmap material build failed ({e!r}); "
                 f"sky dome keeps its default imported material")
    else:
        asset_report["sky"] = "FALLBACK (starmap EXR not available)"
        _log("spawned sky dome WITHOUT real starmap texture (fallback material)")

    if SPACE_LEVEL:
        try:
            _spawn_star_field(subsys, sky.get_actor_location())
            # With point stars in place the textured dome adds nothing but
            # its tessellation seams, so it is hidden in the space level.
            sky.set_actor_hidden_in_game(True)
            asset_report["sky"] = ("procedural point stars (NASA starmap dome hidden: "
                                   "it renders as a dim smear, not stars)")
        except Exception as e:
            _log(f"WARNING: star field not spawned: {e!r}")

    # ── satellite: real NASA ACE model, composite fallback ──
    real_sat_mesh = _import_real_satellite()
    if real_sat_mesh is not None:
        _spawn_satellite_real(START_ENV_POS, real_sat_mesh)
        asset_report["satellite"] = "real (NASA ACE)"
    else:
        _spawn_satellite_composite(START_ENV_POS)
        asset_report["satellite"] = "FALLBACK (composite primitives)"

    # ── goal marker (small bright sphere) ──
    goal = subsys.spawn_actor_from_class(
        unreal.StaticMeshActor, _env_to_world(GOAL_ENV_POS))
    goal.set_actor_label("SafeRL_Goal")
    goal.tags = [GOAL_TAG]
    goal_comp = goal.static_mesh_component
    goal_comp.set_static_mesh(unreal.EditorAssetLibrary.load_asset(SPHERE))
    goal_comp.set_mobility(unreal.ComponentMobility.MOVABLE)
    try:
        goal_comp.set_material(0, _get_or_create_emissive_instance(
            "MI_Goal_Beacon", (0.15, 1.0, 0.35), GOAL_EMISSIVE_GAIN))
        _log(f"goal beacon material applied, gain {GOAL_EMISSIVE_GAIN}")
    except Exception as e:
        _log(f"WARNING: goal beacon material not applied: {e!r}")
    goal.set_actor_scale3d(unreal.Vector(1.5, 1.5, 1.5))
    _log("spawned goal marker")

    # ── debris field: real Poly Haven moon rocks, procedural fallback per-slot ──
    # NUM_DEBRIS actors cycling the scanned meshes. Labels are zero-padded
    # because the live mirror maps actors to published debris in label order,
    # where "Debris_10" would otherwise sort before "Debris_2".
    real_rocks = _import_real_rocks()
    n_real = len(real_rocks)
    asset_report["rocks"] = (f"{NUM_DEBRIS} rocks from {n_real}/"
                             f"{len(MOON_ROCK_NAMES)} real Poly Haven meshes")
    for i in range(NUM_DEBRIS):
        pos = _debris_spawn_pos(i)
        d = subsys.spawn_actor_from_class(
            unreal.StaticMeshActor, _env_to_world(pos))
        d.set_actor_label(f"SafeRL_Debris_{i:02d}")
        d.tags = [DEBRIS_TAG]
        dc = d.static_mesh_component
        dc.set_mobility(unreal.ComponentMobility.MOVABLE)
        k = i % len(DEBRIS_ROTATIONS)
        rx, ry, rz = DEBRIS_ROTATIONS[k]
        # past the first five, turn repeated meshes so they don't read as copies
        rot = unreal.Rotator(rx, ry, rz + (i // len(DEBRIS_ROTATIONS)) * 67.0)

        if n_real:
            dc.set_static_mesh(real_rocks[i % n_real])
            _measure_and_rescale(d, ROCK_TARGET_MAX_DIM_CM[k])
            d.set_actor_rotation(rot, False)
            _log(f"spawned REAL debris {i} (moon_rock) at env {pos}")
        else:
            dc.set_static_mesh(_load_debris_mesh_procedural(k))
            sx, sy, sz = DEBRIS_SCALES[k]
            d.set_actor_scale3d(unreal.Vector(sx, sy, sz))
            d.set_actor_rotation(rot, False)
            _log(f"spawned FALLBACK debris {i} (procedural) at env {pos}")

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
    light_comp.set_editor_property("intensity", SUN_LUX)
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

    # ── camera: becomes the PIE view target, positioned per mode each tick ──
    cam_loc, cam_rot = _camera_pose(CAMERA_MODE)
    cam = subsys.spawn_actor_from_class(unreal.CameraActor, cam_loc, cam_rot)
    cam.set_actor_label("SafeRL_Camera")
    cam.tags = [CAMERA_TAG]

    # ── scene capture for screenshots (follows the camera at capture time) ──
    cap = subsys.spawn_actor_from_class(
        unreal.SceneCapture2D, cam_loc, cam_rot,
    )
    cap.set_actor_label("SafeRL_Capture")
    cap.tags = [CAPTURE_TAG]
    cap.capture_component2d.set_editor_property("capture_every_frame", False)
    cap.capture_component2d.set_editor_property("capture_on_movement", False)

    if SPACE_LEVEL:
        _spawn_space_postfx(subsys, cap)

    unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(
        cam_loc, cam_rot,
    )
    _log(f"scene built (phase 8c real-asset status: {asset_report})")
    asset_report_path = os.path.normpath(os.path.join(_HERE, "..", "..", "phase8c_asset_report.json"))
    try:
        with open(asset_report_path, "w") as f:
            json.dump(asset_report, f, indent=2)
        _log(f"wrote {asset_report_path}")
    except Exception as e:
        _log(f"could not write asset report (non-fatal): {e!r}")


def capture_png(game_world, file_name):
    """Render the live PIE scene through the SceneCapture2D and write a PNG."""
    try:
        actors = unreal.GameplayStatics.get_all_actors_with_tag(game_world, CAPTURE_TAG)
        if not actors:
            _log("no capture actor in the PIE world; skipping screenshot")
            return False
        comp = actors[0].capture_component2d
        cam = _state.get("cam_actor")
        if cam is not None:
            # capture what the viewport is showing, whichever mode is active
            actors[0].set_actor_location_and_rotation(
                cam.get_actor_location(), cam.get_actor_rotation(), False, False)
            comp.set_editor_property(
                "fov_angle", cam.camera_component.get_editor_property("field_of_view"))
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


def _find_debris_in_pie(game_world):
    """Debris actors ordered by index, so index i here is always the same
    piece of debris as index i in the published state. get_all_actors_with_tag
    makes no ordering guarantee. Sorting is numeric on the label's suffix: a
    plain string sort put "Debris_10" before "Debris_2" once there were more
    than ten rocks, silently cross-wiring actors to the wrong published
    positions."""
    actors = unreal.GameplayStatics.get_all_actors_with_tag(game_world, DEBRIS_TAG)

    def index(actor):
        label = actor.get_actor_label()
        try:
            return int(label.rsplit("_", 1)[-1])
        except ValueError:
            return 1 << 30
    return sorted(actors, key=index)


def _setup_view_camera(game_world):
    """Make the camera actor the PIE view target, so the viewport shows the
    same shot the captures take."""
    cams = unreal.GameplayStatics.get_all_actors_with_tag(game_world, CAMERA_TAG)
    if not cams:
        _log("no camera actor in the PIE world; viewport keeps its default view")
        return
    _state["cam_actor"] = cams[0]
    _state["cam_mode"] = CAMERA_MODE
    try:
        pc = unreal.GameplayStatics.get_player_controller(game_world, 0)
        pc.set_view_target_with_blend(cams[0], 0.0)
        _log(f"view target -> {cams[0].get_name()} (camera mode {CAMERA_MODE}); "
             f"switch with: echo chase > {CAMERA_MODE_PATH}")
    except Exception as e:
        _log(f"WARNING: could not set PIE view target: {e!r}")


def _poll_camera_mode():
    try:
        with open(CAMERA_MODE_PATH) as f:
            mode = f.read().strip().lower()
    except Exception:
        return
    if mode in ("wide", "chase", "fixed") and mode != _state.get("cam_mode"):
        _state["cam_mode"] = mode
        _state.pop("chase_pos", None)
        _log(f"camera mode -> {mode}")


def _update_camera():
    cam = _state.get("cam_actor")
    bridge = _state.get("bridge")
    if cam is None or bridge is None:
        return
    mode = _state.get("cam_mode", CAMERA_MODE)
    if mode == "chase":
        sat = bridge.actor.get_actor_location()
        d = _state.get("vel_dir") or (1.0, 0.0, 0.0)
        want = (sat.x - d[0] * CHASE_BACK_CM,
                sat.y - d[1] * CHASE_BACK_CM,
                sat.z - d[2] * CHASE_BACK_CM + CHASE_UP_CM)
        cur = _state.get("chase_pos") or want
        pos = tuple(c + (w - c) * CHASE_LERP for c, w in zip(cur, want))
        _state["chase_pos"] = pos
        loc = unreal.Vector(*pos)
        cam.set_actor_location_and_rotation(
            loc, unreal.MathLibrary.find_look_at_rotation(loc, sat), False, False)
    elif _state.get("cam_applied_mode") != mode:
        loc, rot = _camera_pose(mode)
        cam.set_actor_location_and_rotation(loc, rot, False, False)
    _state["cam_applied_mode"] = mode


def _update_heading(actor, vel):
    """Smoothed velocity direction, for the satellite's heading and the chase
    camera. Visual only: the env has no attitude."""
    if not vel:
        return
    speed = math.sqrt(sum(v * v for v in vel))
    if speed < 0.2:
        return
    v = [c / speed for c in vel]
    old = _state.get("vel_dir")
    if old is None:
        new = v
    else:
        mix = [0.85 * o + 0.15 * n for o, n in zip(old, v)]
        norm = math.sqrt(sum(c * c for c in mix)) or 1.0
        new = [c / norm for c in mix]
    _state["vel_dir"] = new
    if DIMS == 3:
        actor.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(
            unreal.Vector(0.0, 0.0, 0.0), unreal.Vector(*new)), False)


def _tumble_debris(dt):
    """Slow per-rock spin in 3D. Visual only: the env's debris don't rotate."""
    if DIMS != 3:
        return
    for i, actor in enumerate(_state.get("live_debris_actors") or []):
        actor.add_actor_local_rotation(unreal.Rotator(
            ((i * 53) % 40 - 20) * dt, ((i * 29) % 30 - 15) * dt,
            ((i * 71) % 50 - 25) * dt), False, False)


def _live_mirror_tick():
    """Poll the bridge's state file and move UE actors to match.

    Deliberately tolerant: the writer publishes atomically (tmp + rename) but
    this still runs on a different process's schedule, so a missing or
    momentarily unreadable file is normal and must not kill the tick callback.
    """
    st = _state
    try:
        with open(LIVE_STATE_PATH) as f:
            frame = json.load(f)
    except Exception:
        return  # bridge not up yet, or mid-rename; try again next tick

    if frame.get("t") == st.get("live_last_t"):
        return  # no new frame since last tick; nothing to move
    st["live_last_t"] = frame.get("t")

    bridge = st["bridge"]
    agent = frame.get("agent")
    if agent:
        bridge.actor.set_actor_location(_env_to_world(agent), False, False)
    _update_heading(bridge.actor, frame.get("agent_vel"))

    debris_actors = st.get("live_debris_actors") or []
    published = frame.get("debris", [])
    if len(published) != len(debris_actors) and not st.get("live_count_warned"):
        st["live_count_warned"] = True
        _log(f"WARNING: bridge publishes {len(published)} debris but the scene has "
             f"{len(debris_actors)} -- launch both sides with the same config "
             f"(run_ue_demo.sh --config does)")
    if frame.get("dims", DIMS) != DIMS and not st.get("live_dims_warned"):
        st["live_dims_warned"] = True
        _log(f"WARNING: bridge is {frame.get('dims')}D but the scene was built {DIMS}D")
    for i, pos in enumerate(published):
        if i < len(debris_actors):
            debris_actors[i].set_actor_location(_env_to_world(pos), False, False)

    ep = frame.get("episode")
    if ep != st.get("live_last_episode"):
        st["live_last_episode"] = ep
        _log(f"live: episode {ep} started")
    st["live_frames"] = st.get("live_frames", 0) + 1
    if st["live_frames"] % 300 == 0:
        _log(f"live: ep {ep} step {frame.get('ep_step')} "
             f"interventions={frame.get('interventions')} "
             f"reward={frame.get('ep_reward')}")

    # Periodic capture so the run leaves visual evidence of the satellite
    # actually moving, rather than only a position number in a log.
    #
    # Two modes. By default, alternating slots 0/1, so any two consecutive
    # captures can be diffed to show motion without the run filling the disk.
    # With SAFERL_LIVE_CAPTURE_SEQ=N, it instead writes a numbered sequence of
    # N frames plus a sidecar of the metrics at each frame, which is what
    # phase 10 assembles the demo animation from. Capture is expensive -- a
    # SceneCapture2D re-render dragged phase 3's loop from ~119 to ~6
    # steps/sec when left on every frame -- so both modes stay sparse.
    if LIVE_CAPTURE_EVERY and st["live_frames"] % LIVE_CAPTURE_EVERY == 0:
        if LIVE_CAPTURE_SEQ:
            n = st.get("live_seq_n", 0)
            if n < LIVE_CAPTURE_SEQ:
                name = f"live_seq_{n:03d}.png"
                capture_png(bridge.world, name)
                st["live_seq_n"] = n + 1
                st.setdefault("live_seq_meta", []).append({
                    "frame": n, "episode": ep, "ep_step": frame.get("ep_step"),
                    "interventions": frame.get("interventions"),
                    "fallbacks": frame.get("fallbacks"),
                    "ep_reward": frame.get("ep_reward"),
                    "agent": agent, "totals": frame.get("totals"),
                })
                try:
                    meta_path = os.path.normpath(
                        os.path.join(_HERE, "..", "..", "live_seq_meta.json"))
                    with open(meta_path, "w") as f:
                        json.dump(st["live_seq_meta"], f, indent=2)
                except Exception as e:
                    _log(f"live: could not write seq meta (non-fatal): {e!r}")
                _log(f"live: captured {name} ({n+1}/{LIVE_CAPTURE_SEQ}) "
                     f"ep {ep} step {frame.get('ep_step')}")
        else:
            slot = (st["live_frames"] // LIVE_CAPTURE_EVERY) % 2
            capture_png(bridge.world, f"live_policy_capture_{slot}.png")
            _log(f"live: captured live_policy_capture_{slot}.png at "
                 f"ep {ep} step {frame.get('ep_step')} agent={agent}")


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
        _state["dt"] = float(delta_seconds)
        if _state.get("cam_actor") is not None and _state["ticks"] % 10 == 0:
            _poll_camera_mode()

        if _state["phase"] == "boot":
            if _state["ticks"] < BOOT_TICKS:
                return
            # Reentrancy guard: phase 8c's asset import (Interchange glTF/EXR
            # import, MaterialEditingLibrary.recompile_material) pumps
            # Slate's message loop synchronously, which re-fires this same
            # registered tick callback *before* this call returns. Flipping
            # phase to a transitional value here, before doing any of that
            # work, makes a reentrant call hit an unhandled branch and return
            # immediately instead of re-running build_scene() and
            # editor_request_begin_play() again on top of the in-progress
            # call. Found by observation: without this guard, phase 8c's
            # first real run built the scene 12 times and threw an
            # "ObjectInstance is null" exception when an inner call's
            # actor-cleanup destroyed the outer call's freshly spawned sky
            # actor out from under it.
            _state["phase"] = "building"
            _disable_background_throttle()
            if SPACE_LEVEL:
                _ensure_space_level()
            build_scene()
            unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_begin_play()
            _log("PIE requested")
            _state["phase"] = "waiting_for_pie"
            return

        if _state["phase"] == "building":
            # A reentrant tick fired while the outer boot call (above) is
            # still inside build_scene()/editor_request_begin_play(). Do
            # nothing and let the outer call finish and advance the phase.
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

            # Destroy the Pawn PIE auto-spawns at PlayerStart. Not needed by
            # this fully-scripted, non-interactive scene, and removing
            # PlayerStart itself (tried at one point) made PIE hang
            # indefinitely with no spawn point for the GameMode's default
            # Pawn -- so PlayerStart stays, and the Pawn it spawns is culled
            # right after PIE comes up instead.
            try:
                pawns = unreal.GameplayStatics.get_all_actors_of_class(
                    game_world, unreal.Pawn)
                for p in pawns:
                    _log(f"destroying PIE-spawned pawn: {p.get_name()} "
                         f"at {p.get_actor_location()}")
                    p.destroy_actor()
            except Exception as e:
                _log(f"WARNING: could not clean up PIE-spawned pawns "
                     f"(non-fatal): {e!r}")

            _state["bridge"] = PIEBridge(game_world, actor)
            _state["bridge"].reset()
            if UNCAP_FRAMERATE:
                _uncap_framerate(game_world)
            _setup_view_camera(game_world)

            if LIVE_POLICY:
                _state["live_debris_actors"] = _find_debris_in_pie(game_world)
                _log(f"LIVE POLICY MODE: mirroring {LIVE_STATE_PATH}")
                _log(f"live: found {len(_state['live_debris_actors'])} debris actors "
                     f"to mirror")
                _log("live: waiting for the policy bridge to publish frames "
                     "(start saferl.demo.live_policy_bridge if it isn't running)")
                _state["phase"] = "live_mirror"
                return

            _log(f"protocol: {NUM_STEPS} steps, steps_per_tick={STEPS_PER_TICK}, "
                 f"uncapped={UNCAP_FRAMERATE}")
            _state["run_start_tick"] = _state["ticks"]
            _state["t0"] = time.perf_counter()
            _state["phase"] = "running"
            return

        if _state["phase"] == "live_mirror":
            _live_mirror_tick()
            _tumble_debris(_state["dt"])
            _update_camera()
            return

        if _state["phase"] == "posing":
            bridge = _state["bridge"]
            _state["pose_ticks"] += 1
            bridge.step(bridge.action_toward_goal())
            _update_camera()
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

        _update_camera()
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
