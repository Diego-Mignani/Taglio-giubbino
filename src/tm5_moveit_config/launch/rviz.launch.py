from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():

    pkg_path = FindPackageShare("tm_description").find("tm_description")
    urdf_path = os.path.join(pkg_path, "urdf", "tm5-900.urdf")

    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            parameters=[
                {"robot_description": robot_desc},
            ],
        )
    ])
