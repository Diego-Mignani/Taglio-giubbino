from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
from moveit_ros_planning_interface import MoveGroupInterface

class MoveItWrapper:
    def __init__(self, node):
        roscpp_init("tm5_planning", [])
        self.node = node
        self.group = MoveGroupInterface("manipulator")

    def plan_to_pose(self, pose):
        result = self.group.plan_to_pose(pose)
        return result.success, result.trajectory, result.message

    def plan_to_joints(self, joint_state):
        result = self.group.plan_to_joint_state(joint_state)
        return result.success, result.trajectory, result.message
  
    def plan_cartesian_path(self, poses):
        try:
            (traj, fraction) = self.group.compute_cartesian_path(
                poses,
                eef_step=0.01,
                jump_threshold=0.0
            )

            if fraction < 0.9:
                return False, None, f"Cartesian path incomplete: {fraction*100:.1f}%"

            return True, traj.joint_trajectory, "Cartesian path planned successfully"

        except Exception as e:
            return False, None, f"Error in Cartesian planning: {str(e)}"
