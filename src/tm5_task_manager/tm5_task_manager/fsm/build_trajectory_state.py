from .base_state import BaseState
from tm5_trajectory_builder.utils.trajectory_builder import TrajectoryBuilder
from geometry_msgs.msg import Pose

class BuildTrajectoryState(BaseState):

    def on_enter(self):
        self.node.get_logger().info("BUILD_TRAJECTORY → Generating approach + path + retreat")

        # Inizializza il builder
        self.builder = TrajectoryBuilder(self.node)

    def run(self):
        if self.node.path_points is None:
            self.node.get_logger().warn("No path points received yet")
            return "WAIT_TARGET"

        # Home pose dal parametro YAML
        home = Pose()
        home.position.x = self.node.home_pose["position"][0]
        home.position.y = self.node.home_pose["position"][1]
        home.position.z = self.node.home_pose["position"][2]
        home.orientation.x = self.node.home_pose["orientation"][0]
        home.orientation.y = self.node.home_pose["orientation"][1]
        home.orientation.z = self.node.home_pose["orientation"][2]
        home.orientation.w = self.node.home_pose["orientation"][3]

        # Genera la traiettoria completa
        full_path = self.builder.build(self.node.path_points, home)

        # Salva nel Task Manager
        self.node.cartesian_path = full_path

        self.node.get_logger().info(f"Generated full trajectory with {len(full_path)} points")

        return "PLAN"
