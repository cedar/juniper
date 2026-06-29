from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

from ..backend.Exceptions import CircuitConnectionError

from .Configurable import Configurable
from ..frontend import CircuitContext
from ...util import util


logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from .Slot import Slot
    from .Circuit import Circuit

class Connectable(Configurable, ABC):
    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params)
        self.parent_circuit : Circuit = CircuitContext.get_current()

    def __rshift__(self, other : Connectable | str) -> Connectable:
        try:
            other_slot = self.get_slot_from_identifier(other, dir="in" if other is not CircuitContext.get_current() else "out")
            self_slot = self.get_slot_from_identifier(self, dir="out" if self is not CircuitContext.get_current() else "in")
            self._connection_circuit(self_slot, other_slot).connect_to(self_slot, other_slot)
            return other
        except Exception as e:
            raise CircuitConnectionError(f"Connectable::>>: Can't connect '{self.get_path_str()}' to '{other.get_path_str()}' of type {type(other)}") from e

    def __lshift__(self, other : Connectable | str) -> Connectable:
        try:
            other_slot = self.get_slot_from_identifier(other, dir="out" if other is not CircuitContext.get_current() else "in")
            self_slot = self.get_slot_from_identifier(self, dir="in" if self is not CircuitContext.get_current() else "out")
            self._connection_circuit(other_slot, self_slot).connect_to(other_slot, self_slot)
            return self
        except Exception as e:
            raise CircuitConnectionError(f"Connectable::<<: Can't connect '{self.get_path_str()}' to '{other.get_path_str()}' of type ({type(other)})") from e

    def _connection_circuit(self, source: Slot, dest: Slot) -> Circuit:
        """Return the circuit that owns a connection between two slots."""
        current = CircuitContext.get_current()
        if source.parent is current or dest.parent is current:
            return current
        return self.parent_circuit
        
    def get_slot_from_identifier(self, connectable : Connectable | str, dir : str = "out") -> Slot:
        """Is used to retreive a slot from specifier of other connectable. Defaults to the default output slot"""
        other_connectable = self.get_from_identifier(connectable)
        return other_connectable.get_slot(util.DEFAULT_OUTPUT_SLOT) if dir == "out" else other_connectable.get_slot(util.DEFAULT_INPUT_SLOT)
        
    @abstractmethod
    def get_slot(self, slot_id : str):
        """used to get a slot of a conenctable/element"""
        pass

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
        """returns the global path to the connectable as a string. 'circ0.field0'"""
        path = self.get_path()
        path_str = ""
        for sub_str in path:
            path_str += sub_str + "."
        return path_str[:-1]
    
    def get_from_identifier(self, identifier : str | Connectable) -> Connectable:
        """Get a connectable from an identifier (slot, element, path_str)"""
        if isinstance(identifier, str):
            if "." in identifier:
                path_to_element, slot_id = identifier.split(".", maxsplit=1)
                element = self.parent_circuit.get_element(path_to_element)
                slot = element.get_slot(slot_id)
                return slot
            else:
                path_to_element, slot_id = identifier, None
                element = self.parent_circuit.get_element(path_to_element)
                return element
        else:
            return identifier