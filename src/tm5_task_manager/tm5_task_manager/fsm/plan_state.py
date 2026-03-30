from .base_state import BaseState
from tm5_pk_planning.srv import PlanCartesianPath
import rclpy

class PlanState(BaseState):

    def on_enter(self):
        self.node.get_logger().info("PLAN → Planning Cartesian path")

    def run(self):
        if self.node.cartesian_path is None:
            self.node.get_logger().warn("PLAN: No cartesian path available")
            return "WAIT_TARGET"

        req = PlanCartesianPath.Request()
        req.cartesian_path = self.node.cartesian_path

        future = self.node.plan_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

        result = future.result()

        if not result.success:
            self.node.get_logger().error(f"Planning failed: {result.message}")
            return "MOVE_TO_SCAN"

        self.node.current_trajectory = result.trajectory
        self.node.get_logger().info("Planning successful")

        return "EXECUTE"
