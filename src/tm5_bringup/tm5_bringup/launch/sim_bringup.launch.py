from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package="tm5_pk_planning", executable="planning_node"),
        Node(package="tm5_motion_interface", executable="motion_interface", parameters=[{"use_sim": True}]),
        Node(package="tm5_sim_control", executable="sim_controller"),
        Node(package="tm5_task_manager", executable="task_manager"),
    ])
