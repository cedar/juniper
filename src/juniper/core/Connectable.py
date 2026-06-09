from __future__ import annotations
from typing import TYPE_CHECKING
from .Configurable import Configurable
from . import CircuitContext
from ..util import util

if TYPE_CHECKING:
    from .Slot import Slot
    from .Circuit import Circuit

class Connectable(Configurable):
    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params)
        self.parent_circuit : Circuit = CircuitContext.get_current()

    def __rshift__(self, other : Connectable | str) -> Connectable:
        if hasattr(other, "get_slot_id") or hasattr(other, "get_slot") or isinstance(other, str):
            other_slot = self.get_slot_from_connectable(other, dir="in")
            self_slot = self.get_slot_from_connectable(self, dir="out")
            self.parent_circuit.connect_to(self_slot, other_slot)
            return other
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")

    def __lshift__(self, other : Connectable | str) -> Connectable:
        if hasattr(other, "get_slot_id") or hasattr(other, "get_slot") or isinstance(other, str):
            other_slot = self.get_slot_from_connectable(other, dir="out")
            self_slot = self.get_slot_from_connectable(self, dir="in")
            self.parent_circuit.connect_to(other_slot, self_slot)
            return self
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")
        
    def get_slot_from_connectable(self, connectable : Connectable | str, dir : str = "out") -> Slot:
        if hasattr(connectable, "get_slot_id"):
            return connectable
        if hasattr(connectable, "get_slot"):
            return connectable.get_slot(util.DEFAULT_OUTPUT_SLOT) if dir == "out" else connectable.get_slot(util.DEFAULT_INPUT_SLOT)
        if isinstance(connectable, str):
            if "." in connectable:
                other_name, other_slot = connectable.split(".", maxsplit=1)
            else:
                other_name, other_slot = connectable, None
            other_element = self.parent_circuit.get_element(other_name)
            if other_slot is None:
                return other_element.get_slot(util.DEFAULT_OUTPUT_SLOT) if dir == "out" else other_element.get_slot(util.DEFAULT_INPUT_SLOT)
            else:
                return other_element.get_slot(other_slot)
