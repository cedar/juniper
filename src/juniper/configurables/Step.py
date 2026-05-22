from ..util import util
from .Element import Element
from .Buffer import Buffer

class Step(Element):
    def __init__(self, name : str, params : dict, mandatory_params : list, is_dynamic : bool = False):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params)
        self.compute_kernel = None

        if is_dynamic and "shape" not in params.keys():
            raise ValueError(f"Dynamic steps require a shape parameter. ({name})")
        
        self.is_dynamic = is_dynamic
        self.needs_input_connections = True
        self.is_source = False
        self.is_sink = False

        self.buffer_map : dict[str, Buffer] = {}

        # register default step input/output slots
        self.register_input_slot(util.DEFAULT_INPUT_SLOT)
        self.register_output_slot(util.DEFAULT_OUTPUT_SLOT)

        self.parent_circuit.add_element(self)

    def get_buffer(self, buf_id : str) -> Buffer:
        if buf_id in self.buffer_map.keys():
            return self.buffer_map[buf_id]
        else:
            raise Exception(f"Buffer {buf_id} not found ({self.get_name()})")
        
    def register_buffer(self, buf_id : str, shape : tuple, permanent : bool  = False):
        if buf_id in self.buffer_map.keys():
            raise ValueError(f"Buffer {buf_id} already registered in step {self.get_name()}")
        self.buffer_map[buf_id] = Buffer(self, buf_id, shape, permanent)
        
