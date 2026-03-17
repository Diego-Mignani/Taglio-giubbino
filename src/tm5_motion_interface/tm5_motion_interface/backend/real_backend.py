class RealBackend:
    def __init__(self, node):
        self.node = node

    def execute_trajectory(self, request, response):
        # TODO: chiamare FollowJointTrajectory
        response.success = True
        response.message = "Real robot execution"
        return response

    def move_to_pose(self, request, response):
        # TODO: chiamare MoveIt e poi FollowJointTrajectory
        response.success = True
        response.message = "Real robot move"
        return response
