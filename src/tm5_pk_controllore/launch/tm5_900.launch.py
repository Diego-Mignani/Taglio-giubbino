import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Recupera il percorso dell'URDF
    description_package = get_package_share_directory('tm_description')

    # Controlla il nome esatto del file!
    urdf_file = os.path.join(description_package, 'urdf', 'tm5-900.urdf') 
    
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        
        Node(
            package='ros_tcp_endpoint',
            executable='default_server_endpoint',
            name='unity_endpoint',
            parameters=[{'ROS_IP': '0.0.0.0'}]
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
            parameters=[{'robot_description': robot_desc}]
        ),

        Node(
            package='tm5_pk_telecamera',
            executable='punti_telecamera',
            name='tm5_generatore_punti',
            output='screen',
        ),
        
        Node(
            package='tm5_pk_traiettorie',
            executable='genera_traiettorie',
            name='tm5_gestione_traiettoria',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),
    ])


'''''
        Node(
            package='tm5_pk_controllore',
            executable='controlla_robot',
            name='tm5_controller',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),
        '''''