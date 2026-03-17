from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package="tm5_pk_planning", executable="planning_node"),
        Node(package="tm5_motion_interface", executable="motion_interface", parameters=[{"use_sim": False}]),
        Node(package="tm5_robot_control", executable="robot_control"),
        Node(package="tm5_task_manager", executable="task_manager"),
    ])
