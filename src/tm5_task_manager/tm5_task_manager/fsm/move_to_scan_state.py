from .base_state import BaseState
from geometry_msgs.msg import PoseStamped
from tm5_motion_interface.srv import MoveToPose
import rclpy

class MoveToScanState(BaseState):
    def on_enter(self):
        self.node.get_logger().info("MOVE_TO_SCAN → Moving to scan pose")

        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = self.node.scan_pose["position"][0]
        pose.pose.position.y = self.node.scan_pose["position"][1]
        pose.pose.position.z = self.node.scan_pose["position"][2]
        pose.pose.orientation.x = self.node.scan_pose["orientation"][0]
        pose.pose.orientation.y = self.node.scan_pose["orientation"][1]
        pose.pose.orientation.z = self.node.scan_pose["orientation"][2]
        pose.pose.orientation.w = self.node.scan_pose["orientation"][3]

        self.req = MoveToPose.Request()
        self.req.target_pose = pose

    def run(self):
        future = self.node.move_pose_cli.call_async(self.req)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result().success:
            return "WAIT_TARGET"

        self.node.get_logger().error("Failed to reach scan pose")
        return "MOVE_TO_SCAN"
