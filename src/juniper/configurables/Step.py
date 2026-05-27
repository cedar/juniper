from ..util import util
from .Element import Element
from .Buffer import Buffer

class Step(Element):
    def __init__(self, name : str, params : dict, mandatory_params : list):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params)
        
        self.buffer_map : dict[str, Buffer] = {}

        self.parent.add_element(self)

    def get_buffer(self, buf_id : str) -> Buffer:
        if buf_id in self.buffer_map.keys():
            return self.buffer_map[buf_id]
        else:
            raise Exception(f"Buffer {buf_id} not found ({self.get_name()})")
        
    def register_buffer(self, buf_id : str, shape : tuple, permanent : bool  = False):
        if buf_id in self.buffer_map.keys():
            raise ValueError(f"Buffer {buf_id} already registered in step {self.get_name()}")
        self.buffer_map[buf_id] = Buffer(self, buf_id, shape, permanent)

    def check_compiled(self):
        for buffer in self.buffer_map.values():
            if not buffer.is_compiled:
                self.is_compiled=False
                return False
        for slot in self.input_slot_map.values():
            if not slot.is_compiled:
                self.is_compiled=False
                return False
        for slot in self.output_slot_map.values():
            if not slot.is_compiled:
                self.is_compiled=False
                return False

        # all slots and buffer are compiled -> step is compiled
        self.is_compiled = True
        return self.is_compiled       
    
    def compile_state(self, input_slots):
        # Step specific comile function to determine the state of each slot and buffer of the step, given the input slots
        raise NotImplementedError(f"Step missing compile function for state inference ({self.get_name()})")

    def commpile_state(self, input_slots):
        return self.compile_state(input_slots)
