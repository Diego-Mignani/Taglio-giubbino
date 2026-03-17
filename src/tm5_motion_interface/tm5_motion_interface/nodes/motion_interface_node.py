import rclpy
from rclpy.node import Node
from tm5_motion_interface.srv import ExecuteTrajectory, MoveToPose
from tm5_motion_interface.backend.sim_backend import SimBackend
from tm5_motion_interface.backend.real_backend import RealBackend

class MotionInterface(Node):
    def __init__(self):
        super().__init__("tm5_motion_interface")

        self.use_sim = self.declare_parameter("use_sim", True).value
        self.backend = SimBackend(self) if self.use_sim else RealBackend(self)

        self.create_service(ExecuteTrajectory, "execute_trajectory", self.exec_cb)
        self.create_service(MoveToPose, "move_to_pose", self.pose_cb)

    def exec_cb(self, request, response):
        return self.backend.execute_trajectory(request, response)

    def pose_cb(self, request, response):
        return self.backend.move_to_pose(request, response)

def main():
    rclpy.init()
    node = MotionInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
