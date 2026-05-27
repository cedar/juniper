from .Connectable import Connectable
from .Slot import Slot
from ..util import util
from typing import Callable
from typing import Optional
from .Circuit import Circuit

class Element(Connectable):
    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        if "." in name:
            raise ValueError(f"Element names cannot contain dots. ({name})")
        
        super().__init__(name=name,params=params, mandatory_params=mandatory_params)
        self.input_slot_map : dict[str, Slot] = {}
        self.output_slot_map : dict[str, Slot] = {}
        self.compute_kernel : Callable[[dict, dict, Optional[dict]], dict] = None
        self.parent = Circuit.parent_circuit()

        # Element level meta-data for compiling state-info
        self.is_dynamic = False
        self.is_sink = False
        self.is_source = False
        self.manages_sup_process = False
        self.needs_input_connections = True

        # compiler flag to signal that the internal state has been successfully inferred
        self.is_compiled = False

    def register_output_slot(self, slot_id : str):
        if slot_id in self.output_slot_map.keys():
            raise Exception(f"Output slot {slot_id} already registered in step {self.get_name()}")
        slot = Slot(self, slot_id)
        # Register output slot shortcut
        setattr(self, f"{slot_id}", slot)
        # register slot
        self.output_slot_map[slot_id] = slot

    def register_input_slot(self, slot_id : str, max_incoming_connections : int = 1):
        if slot_id in self.input_slot_map.keys():
            raise Exception(f"Input slot {slot_id} already registered in step {self.get_name()}")
        slot = Slot(self, slot_id, max_incoming_connections)
        # Register input slot shortcut
        setattr(self, f"{slot_id}", slot)
        # register slot
        self.input_slot_map[slot_id] = slot

    def get_max_incoming_connections(self, slot_id : str) -> int:
        slot = self.get_slot(slot_id=slot_id)
        return slot.max_incoming_connections
    
    def get_slot(self, slot_id : str) -> Slot:
        try:
            slot = self.get_input_slot(slot_id)
            return slot
        except Exception:
            try:
                slot = self.get_output_slot(slot_id)
                return slot
            except Exception:
                raise Exception(f"Slot {slot_id} does not exist in step {self.get_name()}")
    
    def get_input_slot(self, slot_id : str) -> Slot:
        if slot_id not in self.input_slot_map.keys():
            raise Exception(f"Slot {slot_id} does not exist in step {self.get_name()}")
        return self.input_slot_map[slot_id]
    
    def get_output_slot(self, slot_id : str) -> Slot:
        if slot_id not in self.output_slot_map.keys():
            raise Exception(f"Slot {slot_id} does not exist in step {self.get_name()}")
        return self.output_slot_map[slot_id]
    
    def commpile_state(self, input_slots : dict[str,Slot]) -> bool:
        # Default state inference behavior. No buffer and same shape and dtype of default input slot for default output slot.
        
        return False
