from .base_state import BaseState
from tm5_pk_planning.srv import PlanToPose
import rclpy

class PlanState(BaseState):
    def on_enter(self):
        self.node.get_logger().info("PLAN → Requesting MoveIt plan")

    def run(self):
        if not self.node.plan_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn("MoveIt planning service not available")
            return "PLAN"

        req = PlanToPose.Request()
        req.target_pose = self.node.target_pose

        future = self.node.plan_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

        result = future.result()

        if result.success:
            self.node.current_trajectory = result.trajectory
            return "EXECUTE"

        self.node.get_logger().error("Planning failed: " + result.message)
        return "WAIT_TARGET"
