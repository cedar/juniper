import logging

from .Connectable import Connectable

logger = logging.getLogger(__name__)
class Slot(Connectable):
    def __init__(self, element, slot_id : str, max_incoming_connections : int = 1):
        slot_name = element.get_local_circuit_id() + "." + slot_id
        super().__init__(name=slot_name)
        self.parent = element
        self.slot_id = slot_id
        self.shape = None
        self.dtype = None
        self.max_incoming_connections = max_incoming_connections
        self.is_compiled = False

    @property
    def parent_circuit(self):
        if hasattr(self, "parent"):
            return self.parent.parent_circuit
        return self._parent_circuit

    @parent_circuit.setter
    def parent_circuit(self, circuit):
        self._parent_circuit = circuit

    def get_slot_id(self) -> str:
        return self.slot_id
        
    def get_slot(self, slot_id : str) -> Connectable:
        return self
