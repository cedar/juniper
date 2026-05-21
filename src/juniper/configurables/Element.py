from .Connectable import Connectable
from .Slot import Slot

class Element(Connectable):
    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        if "." in name:
            raise ValueError(f"Element names cannot contain dots. ({name})")
        
        super().__init__(name=name,params=params, mandatory_params=mandatory_params)
        self.input_slot_map = {}
        self.output_slot_map = {}

    def register_output_slot(self, slot_id : str):
        if slot_id in self.output_slot_map.keys():
            raise ValueError(f"Output slot {slot_id} already registered in step {self.get_name()}")
        slot = Slot(self, slot_id)
        # Register output slot shortcut
        setattr(self, f"{slot_id}", slot)
        # register slot
        self.output_slot_map[slot_id] = Slot

    def register_input_slot(self, slot_id : str, max_incoming_connections : int = 1):
        if slot_id in self.input_slot_map.keys():
            raise ValueError(f"Input slot {slot_id} already registered in step {self.get_name()}")
        slot = Slot(self, slot_id, max_incoming_connections)
        # Register input slot shortcut
        setattr(self, f"{slot_id}", slot)
        # register slot
        self.input_slot_map[slot_id] = slot

    def get_max_incoming_connections(self, slot_id : str) -> int:
        slot = self.get_slot(slot_id=slot_id)
        return slot.max_incoming_connections
    
    def get_slot(self, slot_id : str) -> Slot:
        if slot_id not in self.input_slot_map.keys():
            raise ValueError(f"Slot {slot_id} does not exist in step {self.get_name()}")
        return self.input_slot_map[slot_id]
        