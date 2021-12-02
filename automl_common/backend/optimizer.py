from .context import Context

class Optimizer:

    def __init__(self, dir: str, context: Context):
        self.dir = dir
        self.context = context
