from .base_state import BaseState
import time

class WaitTargetState(BaseState):

    def on_enter(self):
        self.node.get_logger().info("WAIT_TARGET → Waiting for camera path")
        self.start_time = time.time()

    def run(self):
        if self.node.path_points is not None:
            return "BUILD_TRAJECTORY"

        if time.time() - self.start_time > self.node.wait_target_timeout:
            self.node.get_logger().warn("WAIT_TARGET timeout")
            return "MOVE_TO_SCAN"

        return "WAIT_TARGET"
