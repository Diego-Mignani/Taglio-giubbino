class SimBackend:
    def __init__(self, node):
        self.node = node

    def execute_trajectory(self, request, response):
        # TODO: inviare traiettoria al PD controller
        response.success = True
        response.message = "Simulated execution"
        return response

    def move_to_pose(self, request, response):
        # TODO: chiamare MoveIt e poi PD controller
        response.success = True
        response.message = "Simulated move"
        return response
