from .base_state import BaseState

class InitState(BaseState):
    def on_enter(self):
        self.node.get_logger().info("INIT → System starting")

    def run(self):
        return "WAIT_TARGET"
