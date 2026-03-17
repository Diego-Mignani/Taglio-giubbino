import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from tm5_mock_camera.utils.zip_generator import generate_zip
from tm5_mock_camera.utils.pocket_generator import generate_pocket

class MockCamera(Node):
    def __init__(self):
        super().__init__("mock_camera")

        self.declare_parameter("mode", "zip")
        self.declare_parameter("publish_rate", 1.0)

        self.mode = self.get_parameter("mode").value
        rate = self.get_parameter("publish_rate").value

        self.pub = self.create_publisher(PoseArray, "/tm5/camera/path", 10)
        self.timer = self.create_timer(1.0 / rate, self.publish_path)

    def publish_path(self):
        msg = PoseArray()
        msg.header.frame_id = "base_link"

        if self.mode == "zip":
            msg.poses = generate_zip(
                start=np.array([0.3, 0.0, 0.15]),
                end=np.array([0.9, 0.5, 0.15])
            )
        elif self.mode == "pocket":
            msg.poses = generate_pocket()
        else:
            msg.poses = generate_zip(...) + generate_pocket(...)

        self.pub.publish(msg)
        self.get_logger().info(f"Published {len(msg.poses)} mock points")

def main():
    rclpy.init()
    node = MockCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
