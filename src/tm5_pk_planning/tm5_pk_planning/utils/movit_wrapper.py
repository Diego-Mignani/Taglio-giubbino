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
