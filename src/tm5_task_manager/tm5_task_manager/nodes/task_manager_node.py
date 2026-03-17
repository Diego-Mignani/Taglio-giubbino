import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from tm5_pk_planning.srv import PlanToPose
from tm5_motion_interface.srv import MoveToPose, ExecuteTrajectory
from geometry_msgs.msg import PoseArray
from tm5_task_manager.fsm.build_trajectory_state import BuildTrajectoryState
from tm5_task_manager.fsm.init_state import InitState
from tm5_task_manager.fsm.move_to_scan_state import MoveToScanState
from tm5_task_manager.fsm.wait_target_state import WaitTargetState
from tm5_task_manager.fsm.plan_state import PlanState
from tm5_task_manager.fsm.execute_state import ExecuteState
from tm5_task_manager.fsm.done_state import DoneState

class TaskManager(Node):
    def __init__(self):
        super().__init__("tm5_task_manager")

        # --- Parametri ---
        self.use_sim = self.declare_parameter("use_sim", True).value
        self.wait_target_timeout = self.declare_parameter("wait_target_timeout", 5.0).value
        self.planning_timeout = self.declare_parameter("planning_timeout", 10.0).value
        self.execution_timeout = self.declare_parameter("execution_timeout", 20.0).value

        self.scan_pose = self.declare_parameter("scan_pose", {}).value

        # --- Variabili ---
        self.target_pose = None
        self.current_trajectory = None

        # --- Client ---
        self.plan_cli = self.create_client(PlanToPose, "plan_to_pose")
        self.exec_cli = self.create_client(ExecuteTrajectory, "execute_trajectory")
        self.move_pose_cli = self.create_client(MoveToPose, "move_to_pose")

        # --- Sottoscrizione telecamera ---
        camera_topic = self.declare_parameter("camera_topic", "/tm5/camera/target_pose").value
        self.create_subscription(PoseStamped, camera_topic, self.target_cb, 10)

        # --- FSM ---
        
        self.states = {
            "INIT": InitState(self),
            "MOVE_TO_SCAN": MoveToScanState(self),
            "WAIT_TARGET": WaitTargetState(self),
            "BUILD_TRAJECTORY": BuildTrajectoryState(self),   # ← NUOVO
            "PLAN": PlanState(self),
            "EXECUTE": ExecuteState(self),
            "DONE": DoneState(self),
        }

        self.current_state_name = "INIT"
        self.current_state = self.states[self.current_state_name]
        self.current_state.on_enter()

        self.timer = self.create_timer(0.1, self.loop)

        self.path_points = None

        self.create_subscription(
            PoseArray,
            "/tm5/camera/path",
            self.path_cb,
            10
        )

    def path_cb(self, msg):
        self.path_points = list(msg.poses)
        self.get_logger().info(f"Received {len(self.path_points)} path points")

    def target_cb(self, msg):
        self.target_pose = msg

    def loop(self):
        next_state = self.current_state.run()

        if next_state != self.current_state_name:
            self.current_state.on_exit()
            self.current_state_name = next_state
            self.current_state = self.states[next_state]
            self.current_state.on_enter()

def main():
    rclpy.init()
    node = TaskManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
