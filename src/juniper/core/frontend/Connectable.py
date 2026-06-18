from __future__ import annotations
from typing import TYPE_CHECKING
from .Configurable import Configurable
from ..frontend import CircuitContext
from ...util import util

if TYPE_CHECKING:
    from .Slot import Slot
    from .Circuit import Circuit

class Connectable(Configurable):
    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params)
        self.parent_circuit : Circuit = CircuitContext.get_current()

    def __rshift__(self, other : Connectable | str) -> Connectable:
        if hasattr(other, "get_slot_id") or hasattr(other, "get_slot") or isinstance(other, str):
            other_slot = self.get_slot_from_connectable(other, dir="in" if other is not CircuitContext.get_current() else "out")
            self_slot = self.get_slot_from_connectable(self, dir="out" if self is not CircuitContext.get_current() else "in")
            self._connection_circuit(self_slot, other_slot).connect_to(self_slot, other_slot)
            return other
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")

    def __lshift__(self, other : Connectable | str) -> Connectable:
        if hasattr(other, "get_slot_id") or hasattr(other, "get_slot") or isinstance(other, str):
            other_slot = self.get_slot_from_connectable(other, dir="out" if other is not CircuitContext.get_current() else "in")
            self_slot = self.get_slot_from_connectable(self, dir="in" if self is not CircuitContext.get_current() else "out")
            self._connection_circuit(other_slot, self_slot).connect_to(other_slot, self_slot)
            return self
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")

    def _connection_circuit(self, source: Slot, dest: Slot) -> Circuit:
        """Return the circuit that owns a connection between two slots."""
        current = CircuitContext.get_current()
        if source.parent is current or dest.parent is current:
            return current
        return self.parent_circuit
        
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
            
    def get_path(self) -> tuple[str,...]:
        """returns the global path to the connectable as a tuple of strings. ('circ0', 'field0')"""
        obj_path = []
        current = self

        while True:
            parent = current.parent_circuit
            if parent is None or parent is current:
                break
            obj_path.insert(0, current)
            current = parent

        return tuple(obj.get_local_circuit_id() for obj in obj_path)
    
    def get_path_str(self) -> str:
        """returns the global path to the connectable as a string. 'circ0.field0')"""
        path = self.get_path()
        path_str = ""
        for sub_str in path:
            path_str += sub_str + "."
        return path_str[:-1]
