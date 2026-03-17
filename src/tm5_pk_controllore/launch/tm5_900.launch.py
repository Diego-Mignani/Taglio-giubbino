import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import yaml

def load_yaml(package_name, file_path):
    pkg_path = get_package_share_directory(package_name)
    abs_file_path = os.path.join(pkg_path, file_path)
    with open(abs_file_path, 'r') as f:
        return yaml.safe_load(f)

def generate_launch_description():
    # URDF dal package tm_description
    description_package = get_package_share_directory('tm_description')
    urdf_file = os.path.join(description_package, 'urdf', 'tm5-900.urdf')
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # YAML guadagni controller
    controller_pkg = get_package_share_directory('tm5_pk_controllore')
    gains_yaml = os.path.join(controller_pkg, 'config', 'controller_gains.yaml')

    # MoveIt: includo il launch originale del pacchetto
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("tm5_moveit_config"),
                "launch",
                "move_group.launch.py"
            ])
        )
    )

    # Robot State Publisher (serve sia a RViz che a tutto il resto)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    return LaunchDescription([
        moveit_launch,
        robot_state_publisher_node,

        Node(
            package='ros_tcp_endpoint',
            executable='default_server_endpoint',
            name='unity_endpoint',
            parameters=[{'ROS_IP': '0.0.0.0'}],
            output='screen'
        ),

        Node(
            package="tm5_pk_controllore",
            executable="tm5_joint_state_bridge",
            output="screen"
        ),

        Node(
            package='tm5_pk_controllore',
            executable='comunica_robot',
            name='tm5_unity_comunicator',
            output='screen',
        ),

        Node(
            package='tm5_pk_controllore',
            executable='controlla_robot',
            name='tm5_robot_controller',
            output='screen',
            parameters=[
                gains_yaml,
                {'robot_description': robot_desc}
            ]
        ),

        Node(
            package='tm5_pk_telecamera',
            executable='punti_telecamera',
            name='tm5_generatore_punti',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),

        Node(
            package='tm5_pk_traiettorie',
            executable='genera_traiettorie',
            name='tm5_gestione_traiettoria',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),
    ])
