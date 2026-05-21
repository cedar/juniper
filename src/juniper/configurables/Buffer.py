from .Configurable import Configurable
from .Step import Step

class Buffer(Configurable):
    def __init__(self, step : Step, buffer_id : str, shape : tuple, permanent : bool = False):
        name = step.get_name() + "." + buffer_id
        super().__init__(name=name)
        self._parent = step
        self._shape = shape
        self._permanent = permanent

    def get_buffer_id(self) -> str:
        return self.get_name().split(".")[0]