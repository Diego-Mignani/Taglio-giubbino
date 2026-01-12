import math
import rclpy
from std_msgs.msg import Float64MultiArray
from rclpy.node import Node
from sensor_msgs.msg import JointState

class UnityComunicator(Node):
    def __init__(self):
        super().__init__('tm5_unity_comunicator')

        # Canali di comunicazione Ros2 <-> Unity
        self.subscription = self.create_subscription(Float64MultiArray, 'joint_commands', self.listener_publisher_callback, 10)
        self.publisher_ = self.create_publisher(JointState, 'unity_in_joint_states', 10)
        
        # Nomi dei giunti del robot
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']


    def listener_publisher_callback(self, msg):
        '''
        Callback che riceve i comandi di giunto dal controller e li pubblica a Unity.

        Args:
            msg (Float64MultiArray): Messaggio contenente i comandi di giunto
        '''

        # Controllo sicurezza: verifica che arrivino 6 valori
        self.get_logger().info(f"Ricevuto comando da controller: {msg.data[0]}")
        if len(msg.data) < 6:
            self.get_logger().warn(f"Ricevuti solo {len(msg.data)} giunti, previsti 6")
            return
    
        # Creazione del messaggio JointState
        output = JointState()
        
        # Aggiungi il Timestamp (fondamentale per Unity)
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = 'base_link' # o il nome della tua base
        
        # Assegna nomi e posizioni
        output.name = self.joint_names
        output.position = list(msg.data)

        # Invia a Unity tramite ROS-TCP-Endpoint
        self.publisher_.publish(output)


def main(args=None):
    rclpy.init(args=args)
    nodo = UnityComunicator()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


"""
def pubblica(self):
        self.t += 0.05
        movimento = math.sin(self.t)
        output = JointState()
        output.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        output.position = [movimento * 0.5, -0.2, movimento * 1.2, 0.0, 0.5, 0.0]
        self.publisher_.publish(output)
        self.get_logger().info(f'Pubblicato movimento: {movimento:.2f}')
"""