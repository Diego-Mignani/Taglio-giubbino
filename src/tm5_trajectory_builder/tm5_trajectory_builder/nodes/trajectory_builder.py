import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from tm5_trajectory_builder.utils.trajectory_builder import TrajectoryBuilder

class TrajectoryBuilderNode(Node):
    def __init__(self):
        super().__init__("trajectory_builder_test")

        self.declare_parameter("approach_steps", 30)
        self.declare_parameter("retreat_steps", 30)
        self.declare_parameter("path_steps", 10)

        self.builder = TrajectoryBuilder(self)

        # Test: zip fittizia
        points = []
        for x in [0.4, 0.45, 0.5, 0.55]:
            p = Pose()
            p.position.x = x
            p.position.y = 0.1
            p.position.z = 0.3
            p.orientation.w = 1.0
            points.append(p)

        # Home pose
        home = Pose()
        home.position.x = 0.3
        home.position.y = 0.0
        home.position.z = 0.5
        home.orientation.w = 1.0

        full = self.builder.build(points, home)

        self.get_logger().info(f"Generated {len(full)} trajectory points")

def main():
    rclpy.init()
    node = TrajectoryBuilderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
