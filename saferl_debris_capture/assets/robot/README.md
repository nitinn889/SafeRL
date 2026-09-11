# Assets directory
# Place robot URDF/USD files here

# Recommended robot models:
# - Franka Panda (7-DOF): available from https://github.com/nicholasmr/franka_panda_description
# - Universal Robots UR5e: available from https://github.com/UniversalRobots/Universal_Robots_ROS2_Description
# - Custom spacecraft-mounted arm: design in Fusion 360 or Onshape → export as URDF

# For Isaac Lab, use USD format:
# Convert URDF → USD using:
#   ./isaaclab.sh -p scripts/tools/convert_urdf.py <path/to/robot.urdf> <output/robot.usd>

# File naming convention:
#   franka_panda.usd    - 7-DOF Franka Panda manipulator
#   ur5e.usd            - 6-DOF Universal Robots UR5e
#   gripper.usd         - custom debris-capture gripper end-effector
