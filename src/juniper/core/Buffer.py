from .Configurable import Configurable

class Buffer(Configurable):
    def __init__(self, step, buffer_id : str, shape : tuple, permanent : bool = False, dtype = None):
        name = buffer_id
        super().__init__(name=name)
        self.parent = step
        self.shape = shape
        self.dtype = dtype
        self.permanent = permanent
        self.is_compiled = False

    def get_buffer_id(self) -> str:
        return self.get_name()
    
