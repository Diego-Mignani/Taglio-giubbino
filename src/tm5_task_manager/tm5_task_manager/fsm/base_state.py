class BaseState:
    def __init__(self, node):
        self.node = node

    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def run(self):
        raise NotImplementedError
