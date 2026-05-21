from .Connectable import Connectable
from .Element import Element

class Slot(Connectable):
    def __init__(self, element : Element, slot_id : str, max_incoming_connections : int = 1):
        slot_name = element.get_name() + "." + slot_id
        self._parent = element
        self.max_incoming_connections = max_incoming_connections

        super().__init__(name=slot_name)

    def get_slot_id(self) -> str:
        return self.get_name().split(".")[1]
        