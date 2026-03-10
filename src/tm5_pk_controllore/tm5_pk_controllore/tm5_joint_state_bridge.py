import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateBridge(Node):
    def __init__(self):
        super().__init__("tm5_joint_state_bridge")

        # ⚠️ METTI QUI I NOMI DEI GIUNTI DEL TUO URDF
        self.urdf_joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]

        self.sub = self.create_subscription(
            JointState,
            "unity_joint_feedback",
            self.cb,
            10
        )

        self.pub = self.create_publisher(
            JointState,
            "joint_states",
            10
        )

    def cb(self, msg):
        # Rinomina i giunti
        msg.name = self.urdf_joint_names
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = JointStateBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
