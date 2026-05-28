from .Configurable import Configurable

class Buffer(Configurable):
    def __init__(self, step, buffer_id : str, shape : tuple, permanent : bool = False):
        name = step.get_name() + "." + buffer_id
        super().__init__(name=name)
        self.parent = step
        self.shape = shape
        self.dtype = None
        self.permanent = permanent
        self.is_compiled = False

    def get_buffer_id(self) -> str:
        return self.get_name().split(".", 1)[1]
    
    def check_compiled(self):
        if self.shape is not None and self.dtype is not None and self.permanent is not None:
            self.is_compiled = True
        return self.is_compiled
