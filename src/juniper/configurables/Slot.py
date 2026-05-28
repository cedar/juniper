from .Connectable import Connectable

class Slot(Connectable):
    def __init__(self, element, slot_id : str, max_incoming_connections : int = 1):
        slot_name = element.get_name() + "." + slot_id
        super().__init__(name=slot_name)
        self.parent = element
        self.shape = None
        self.dtype = None
        self.max_incoming_connections = max_incoming_connections
        self.is_compiled = False


    def get_slot_id(self) -> str:
        return self.get_name().split(".")[1]
        
    def check_compiled(self):
        if self.shape is not None and self.dtype is not None:
            self.is_compiled = True
        return self.is_compiled
