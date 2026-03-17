from .base_state import BaseState
from tm5_motion_interface.srv import ExecuteTrajectory
import rclpy

class ExecuteState(BaseState):
    def on_enter(self):
        self.node.get_logger().info("EXECUTE → Sending trajectory to Motion Interface")

    def run(self):
        if self.node.current_trajectory is None:
            return "WAIT_TARGET"

        req = ExecuteTrajectory.Request()
        req.trajectory = self.node.current_trajectory

        future = self.node.exec_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

        result = future.result()

        if result.success:
            return "DONE"

        self.node.get_logger().error("Execution failed: " + result.message)
        return "WAIT_TARGET"
