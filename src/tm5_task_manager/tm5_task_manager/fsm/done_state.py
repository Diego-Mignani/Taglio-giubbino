from .base_state import BaseState

class DoneState(BaseState):
    def on_enter(self):
        self.node.get_logger().info("DONE → Task completed successfully")

    def run(self):
        return "DONE"
