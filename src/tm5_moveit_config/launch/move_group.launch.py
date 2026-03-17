from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml
import os

def load_yaml(package_name, file_path):
    pkg_path = FindPackageShare(package_name).find(package_name)
    abs_file_path = os.path.join(pkg_path, file_path)
    with open(abs_file_path, 'r') as f:
        return yaml.safe_load(f)

def generate_launch_description():

    # Carica URDF come stringa
    pkg_path = FindPackageShare("tm_description").find("tm_description")
    urdf_path = os.path.join(pkg_path, "urdf", "tm5-900.urdf")
    with open(urdf_path, 'r') as infp:
        robot_description = infp.read()

    robot_description_semantic = load_yaml(
        "tm5_moveit_config",
        "config/tm5.srdf"
    )

    kinematics_yaml = load_yaml(
        "tm5_moveit_config",
        "config/kinematics.yaml"
    )

    ompl_yaml = load_yaml(
        "tm5_moveit_config",
        "config/ompl_planning.yaml"
    )

    joint_limits_yaml = load_yaml(
        "tm5_moveit_config",
        "config/joint_limits.yaml"
    )

    return LaunchDescription([
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                {"robot_description": robot_description},
                {"robot_description_semantic": robot_description_semantic},
                {"robot_description_kinematics": kinematics_yaml},
                {"robot_description_planning": ompl_yaml},
                {"robot_description_planning_scene": joint_limits_yaml},
                {"move_group.planning_plugin": "ompl_interface/OMPLPlanner"},
            ],
        )
    ])
