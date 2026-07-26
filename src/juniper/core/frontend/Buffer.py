import logging
from .Configurable import Configurable


logger = logging.getLogger(__name__)
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
        return self.get_local_circuit_id()
    
    def get_path(self) -> tuple[str,...]:
        """returns the global path to the connectable as a tuple of strings. ('circ0', 'field0')"""
        obj_path = []
        current = self

        while True:
            parent = current.parent_circuit
            if parent is None or parent is current:
                break
            obj_path.insert(0, current)
            current = parent

        return tuple(obj.get_local_circuit_id() for obj in obj_path)
    
    def get_path_str(self) -> str:
        """returns the global path to the connectable as a string. 'circ0.field0'"""
        path = self.get_path()
        path_str = ""
        for sub_str in path:
            path_str += sub_str + "."
        return path_str[:-1]