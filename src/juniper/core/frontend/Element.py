from __future__ import annotations
from typing import Any
from .Connectable import Connectable
from .Slot import Slot
from typing import Callable
from typing import Optional

class Element(Connectable):
    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        if "." in name:
            raise ValueError(f"Element names cannot contain dots. ({name})")
        
        super().__init__(name=name,params=params, mandatory_params=mandatory_params)
        self.input_slot_map : dict[str, Slot] = {}
        self.output_slot_map : dict[str, Slot] = {}
        self.compute_kernel : Callable[[dict, dict, Optional[dict]], dict] = None
        self.parent = self.parent_circuit

        # Element level meta-data for compiling state-info
        self.is_dynamic = False
        self.is_sink = False
        self.is_source = False
        self.needs_input_connections = True
        self.input_aggregation = "sum"

        # compiler flag to signal that the internal state has been successfully inferred
        self.is_compiled = False

    @classmethod
    def from_params(cls, name : str, params : dict[str, Any]) -> Any:
        """Init element from param dict"""
        return cls(name=name, **params)

    def register_output_slot(self, slot_id : str) -> Slot:
        if slot_id in self.output_slot_map.keys():
            raise Exception(f"Output slot {slot_id} already registered in step {self.get_local_circuit_id()}")
        slot = Slot(self, slot_id)
        # Register output slot shortcut
        setattr(self, f"{slot_id}", slot)
        # register slot
        self.output_slot_map[slot_id] = slot
        return slot

    def register_input_slot(self, slot_id : str, max_incoming_connections : int = 1) -> Slot:
        if slot_id in self.input_slot_map.keys():
            raise Exception(f"Input slot {slot_id} already registered in step {self.get_local_circuit_id()}")
        slot = Slot(self, slot_id, max_incoming_connections)
        # Register input slot shortcut
        if getattr(self, f"{slot_id}", None) is None:
            setattr(self, f"{slot_id}", slot)
        else:
            raise Exception(f"{slot_id} is already registered in step {self.get_local_circuit_id()}")
        # register slot
        self.input_slot_map[slot_id] = slot
        # register input slot with parent circuit if not already done so
        if slot.get_local_circuit_id() not in self.parent_circuit.connection_map_reversed.keys():
            self.parent_circuit.connection_map_reversed[slot.get_local_circuit_id()] = []
        return slot

    def get_max_incoming_connections(self, slot_id : str) -> int:
        slot = self.get_slot(slot_id=slot_id)
        return slot.max_incoming_connections
    
    def set_max_incoming_connections(self, slot_id : str, max_incoming_connections : int):
        slot = self.get_slot(slot_id=slot_id)
        slot.max_incoming_connections = max_incoming_connections
    
    def get_slot(self, slot_id : str) -> Slot:
        try:
            slot = self.get_input_slot(slot_id)
            return slot
        except Exception:
            try:
                slot = self.get_output_slot(slot_id)
                return slot
            except Exception:
                raise Exception(f"Slot {slot_id} does not exist in step {self.get_local_circuit_id()}")
    
    def get_input_slot(self, slot_id : str) -> Slot:
        if slot_id not in self.input_slot_map.keys():
            raise Exception(f"Slot {slot_id} does not exist in step {self.get_local_circuit_id()}")
        return self.input_slot_map[slot_id]
    
    def get_output_slot(self, slot_id : str) -> Slot:
        if slot_id not in self.output_slot_map.keys():
            raise Exception(f"Slot {slot_id} does not exist in step {self.get_local_circuit_id()}")
        return self.output_slot_map[slot_id]
    
