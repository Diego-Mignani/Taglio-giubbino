import rclpy
from rclpy.node import Node
from tm5_pk_planning.srv import PlanToPose, PlanToJoints
from tm5_pk_planning.utils.moveit_wrapper import MoveItWrapper
from tm5_pk_planning.srv import PlanCartesianPath


class PlanningNode(Node):
    def __init__(self):
        super().__init__("tm5_planning")

        self.moveit = MoveItWrapper(self)

        self.create_service(PlanToPose, "plan_to_pose", self.plan_pose_cb)
        self.create_service(PlanToJoints, "plan_to_joints", self.plan_joints_cb)
        self.create_service(
            PlanCartesianPath,
            "plan_cartesian_path",
            self.plan_cartesian_cb
        )

    def plan_cartesian_cb(self, request, response):
        poses = request.cartesian_path

        # Usa MoveItWrapper per pianificare una traiettoria cartesiana reale
        success, traj, msg = self.moveit.plan_cartesian_path(poses)

        response.success = success
        response.trajectory = traj
        response.message = msg

        return response


    def plan_pose_cb(self, request, response):
        success, traj, msg = self.moveit.plan_to_pose(request.target_pose)
        response.success = success
        response.trajectory = traj
        response.message = msg
        return response

    def plan_joints_cb(self, request, response):
        success, traj, msg = self.moveit.plan_to_joints(request.target_joints)
        response.success = success
        response.trajectory = traj
        response.message = msg
        return response
    


def main():
    rclpy.init()
    node = PlanningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
