# Assets directory
# Place debris USD model files here

# Debris model requirements:
# - Rigid body, no articulation needed
# - Approximate geometry: box or cylinder (real satellites are box-shaped)
# - Mass properties: configurable via USD physics properties
# - No texture required (simulation only), but add for visualisation

# Recommended sources:
# - NASA 3D resources: https://nasa3d.arc.nasa.gov/
# - ESA CAD models (contact ESA for licensed models)
# - Custom parameterised model: generate from debris_dynamics.py inertia params

# File naming convention:
#   debris_satellite.usd   - primary debris model (default)
#   debris_rocket_body.usd - rocket upper stage debris
#   debris_panel.usd       - solar panel fragment

# To create a simple box debris in USD:
# from pxr import Usd, UsdGeom, UsdPhysics
# stage = Usd.Stage.CreateNew("debris_satellite.usd")
# xform = UsdGeom.Xform.Define(stage, "/Debris")
# cube = UsdGeom.Cube.Define(stage, "/Debris/Body")
# cube.GetSizeAttr().Set(2.0)  # 2m x 2m x 2m box
# physicsAPI = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
# stage.Save()
