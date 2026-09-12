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
SATELLITE_SRC_DIR = os.path.join(MESH_DIR, "satellite")
ROCKS_SRC_DIR = os.path.join(MESH_DIR, "rocks")
SKYBOX_SRC_DIR = os.path.join(MESH_DIR, "skybox")

UNCAP_FRAMERATE = os.environ.get("SAFERL_UNCAP") == "1"
STEPS_PER_TICK = int(os.environ.get("SAFERL_STEPS_PER_TICK", "1"))
RESULTS_NAME = os.environ.get("SAFERL_RESULTS_NAME", "pie_session_results.json")
RESULTS_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", RESULTS_NAME))

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


def _get_or_create_starmap_material(texture):
    """Build (or reuse) an unlit emissive material that shows `texture`
    directly regardless of scene lighting -- the standard technique for a
    textured skybox dome."""
    content_path = f"{TEXTURES_CONTENT_PATH}/M_Starmap_Sky"
    existing = unreal.EditorAssetLibrary.load_asset(content_path)
    if existing is not None:
        return existing

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    factory = unreal.MaterialFactoryNew()
    material = asset_tools.create_asset(
        "M_Starmap_Sky", TEXTURES_CONTENT_PATH, unreal.Material, factory)

    tex_expr = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureSample, -350, 0)
    tex_expr.texture = texture

    try:
        material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    except Exception as e:
        _log(f"could not set unlit shading model (non-fatal): {e!r}")

    unreal.MaterialEditingLibrary.connect_material_property(
        tex_expr, "RGB", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    unreal.MaterialEditingLibrary.recompile_material(material)
    unreal.EditorAssetLibrary.save_asset(content_path)
    _log(f"built unlit starmap material: {content_path}")
    return material


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

    # import phase 8 procedural OBJ meshes (used as fallback source only)
    _import_obj_meshes()

    # ── sky dome: real NASA starmap texture on the inverted-dome geometry,
    #    falls back to phase 8's flat dark material if the EXR/material
    #    pipeline fails ──
    sky_mesh = _load_sky_dome_mesh_procedural()
    sky = subsys.spawn_actor_from_class(
        unreal.StaticMeshActor,
        unreal.Vector(ENV_SIZE * SCALE / 2, ENV_SIZE * SCALE / 2, 0),
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
    goal.set_actor_scale3d(unreal.Vector(1.5, 1.5, 1.5))
    _log("spawned goal marker")

    # ── debris field: real Poly Haven moon rocks, procedural fallback per-slot ──
    real_rocks = _import_real_rocks()
    n_real = len(real_rocks)
    asset_report["rocks"] = f"{n_real}/5 real (Poly Haven moon_rock)"
    for i, pos in enumerate(DEBRIS_ENV_POS):
        d = subsys.spawn_actor_from_class(
            unreal.StaticMeshActor, _env_to_world(pos))
        d.set_actor_label(f"SafeRL_Debris_{i}")
        d.tags = [DEBRIS_TAG]
        dc = d.static_mesh_component
        dc.set_mobility(unreal.ComponentMobility.MOVABLE)

        if i < n_real:
            mesh_obj = real_rocks[i]
            dc.set_static_mesh(mesh_obj)
            _measure_and_rescale(d, ROCK_TARGET_MAX_DIM_CM[i])
            rx, ry, rz = DEBRIS_ROTATIONS[i]
            d.set_actor_rotation(unreal.Rotator(rx, ry, rz), False)
            _log(f"spawned REAL debris {i} (moon_rock) at env {pos}")
        else:
            debris_mesh = _load_debris_mesh_procedural(i)
            dc.set_static_mesh(debris_mesh)
            sx, sy, sz = DEBRIS_SCALES[i]
            d.set_actor_scale3d(unreal.Vector(sx, sy, sz))
            rx, ry, rz = DEBRIS_ROTATIONS[i]
            d.set_actor_rotation(unreal.Rotator(rx, ry, rz), False)
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
