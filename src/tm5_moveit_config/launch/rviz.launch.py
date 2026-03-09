from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    pkg_share = FindPackageShare("tm5_moveit_config")

    robot_description = PathJoinSubstitution(
        [FindPackageShare("tm_description"), "urdf", "tm5-900.urdf"]
    )

    return LaunchDescription([
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            parameters=[
                {"robot_description": Command([FindExecutable(name="xacro"), " ", robot_description])},
            ],
        )
    ])

