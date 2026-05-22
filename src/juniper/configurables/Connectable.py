from __future__ import annotations
from .Configurable import Configurable
from .Slot import Slot
from .Circuit import Circuit
from .Step import Step
from ..util import util

class Connectable(Configurable):
    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params)
        self.parent_circuit : Circuit = Circuit.parent_circuit()

    def __rshift__(self, other : Connectable | str) -> Connectable:
        if isinstance(other, Step) or isinstance(other, Slot) or isinstance(other, Circuit) or isinstance(other, str):
            other_slot = self.get_slot_from_connectable(other, dir="in")
            self.parent_circuit.connect_to(self, other_slot)
            return other
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")

    def __lshift__(self, other : Connectable | str) -> Connectable:
        if isinstance(other, Step) or isinstance(other, Slot) or isinstance(other, Circuit):
            other_slot = self.get_slot_from_connectable(other)
            self.parent_circuit.connect_to(other_slot, self)
            return self
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")
        
    def get_slot_from_connectable(self, connectable : Connectable | str, dir : str = "out") -> Slot:
        if isinstance(connectable, Slot):
            return connectable
        if isinstance(connectable, Step) or isinstance(connectable, Circuit):
            return connectable.get_slot(util.DEFAULT_OUTPUT_SLOT) if dir == "out" else connectable.get_slot(util.DEFAULT_INPUT_SLOT)
        if isinstance(connectable, str):
            other_name, other_slot = connectable.split('.')
            other_element = self.parent_circuit.get_element(other_name)
            if other_slot is None:
                return other_element.get_slot(util.DEFAULT_OUTPUT_SLOT) if dir == "out" else other_element.get_slot(util.DEFAULT_INPUT_SLOT)
            else:
                return other_element.get_slot(other_slot)
