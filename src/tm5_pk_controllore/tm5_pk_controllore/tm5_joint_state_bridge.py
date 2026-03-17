import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateBridge(Node):
    def __init__(self):
        super().__init__("tm5_joint_state_bridge")

        # Mappa: Unity → URDF
        self.unity_to_urdf = {
            "link_1": "joint_1",
            "link_2": "joint_2",
            "link_3": "joint_3",
            "link_4": "joint_4",
            "link_5": "joint_5",
            "link_6": "joint_6",
        }

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
        # Debug per sicurezza
        print("DEBUG Unity joint names:", msg.name)
        print("DEBUG Unity joint positions:", msg.position)

        # Dizionario nome → posizione
        name_to_pos = dict(zip(msg.name, msg.position))

        # Costruisco il JointState corretto
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = "base"   # ⚠️ Il tuo URDF usa "base" come radice

        # Ordine URDF
        js.name = list(self.unity_to_urdf.values())

        # Riordino le posizioni secondo l’ordine URDF
        js.position = [name_to_pos[unity_name] for unity_name in self.unity_to_urdf.keys()]

        self.pub.publish(js)


def main():
    rclpy.init()
    node = JointStateBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
