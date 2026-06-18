from __future__ import annotations
from typing import Callable
from typing import Optional

from .Slot import Slot
from .Element import Element
from . import CircuitContext

def compute_kernel_factory(output_slot_map : dict[str,Slot]) -> Callable[[dict, dict, Optional[dict]], dict]:
    def compute_kernel(input : dict, state : dict, **kwargs : Optional[dict]) -> dict:
        """
            - input: {"out0":jax_array, "ou1":jax_array, ...}
            - state: {"out0":jax_array, "ou1":jax_array, ...}
"""

        out = state.copy()
        for slot_id in output_slot_map.keys():
            out[slot_id] = input[slot_id]
        return out

    return compute_kernel

class Circuit(Element):
    _current : Circuit | None = None

    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params)
        self.element_map : dict[str,Element] = {}
        self.connection_map_reversed : dict[str, list[Slot]] = {}

    @classmethod
    def parent_circuit(cls : Circuit) -> Circuit | None:
        if CircuitContext.get_current() is None:
            raise RuntimeError("No active circuit. This should never happen.")
        return CircuitContext.get_current()
    
    def __enter__(self) -> Circuit:
        self._previous_circuit = CircuitContext.get_current()
        Circuit._current = self
        CircuitContext.set_current(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.define_circuit_structure()
        if self.parent is not self:
            self.parent.add_element(self)
        Circuit._current = self._previous_circuit
        CircuitContext.set_current(self._previous_circuit)
        return False
    
    def clean(self) -> None:
        """Deletes all elements and slots in the circuit."""
        self.element_map = {}
        self.connection_map_reversed = {}
        if self.get_local_circuit_id() in self.parent.element_map.keys():
            self.parent.element_map.pop(self.get_local_circuit_id())
        for slot in self.input_slot_map.values():
            if slot.get_local_circuit_id() in self.parent.connection_map_reversed.keys():
                self.parent.connection_map_reversed.pop(slot.get_local_circuit_id())
        for slot in self.output_slot_map.values():
            if slot.get_local_circuit_id() in self.parent.connection_map_reversed.keys():
                self.parent.connection_map_reversed.pop(slot.get_local_circuit_id())

        self.is_compiled = False
        self.compute_kernel = None


    def add_element(self, element : Element):
        element_name = element.get_local_circuit_id()
        if element_name in self.element_map.keys():
            raise Exception(f"Circuit::add_element(): Element {element_name} already exists in Circuit ({self.get_local_circuit_id()})")
        if self is element:
            raise Exception(f"Circuit::add_element(): A Circuit can't be added as a sub-element to itself ({self.get_local_circuit_id()}).")
        self.element_map[element_name] = element
        for slot in element.input_slot_map.values():
            #print(slot.get_local_circuit_id())
            self.connection_map_reversed[slot.get_local_circuit_id()] = []
    
    def get_elements(self) -> dict[str,Element]:
        return self.element_map
    
    def get_element(self, name : str) -> Element:
        if name not in self.element_map:
            raise Exception(f"Architecture::get_element(): Element {name} not found in Architecture")
        return self.element_map[name]
    
    def get_incoming_elements(self, dest : Element) -> list[Slot]:
        incoming_steps = []
        if isinstance(dest, Slot):
            incoming_steps = self.connection_map_reversed[dest.get_local_circuit_id()]
        else:
            for slot in dest.input_slot_map.values():
                incoming_steps += self.connection_map_reversed[slot.get_local_circuit_id()]
        return incoming_steps
    
    def connect_to(self, source : Slot, dest : Slot):
        """Connect 2 slots."""
        if not isinstance(source, Slot) or not isinstance(dest, Slot):
            source = self.get_slot_from_connectable(source)
            dest = self.get_slot_from_connectable(dest)
        source_name = source.get_local_circuit_id()
        dest_name = dest.get_local_circuit_id()

        if source in self.connection_map_reversed[dest_name]:
            raise Exception(f"Circuit::connect_to(): Connection from {source_name} to {dest_name} already exists ({self.get_local_circuit_id()})")
        
        if len(self.connection_map_reversed[dest_name]) >= dest.max_incoming_connections:
            raise Exception(f"Circuit::connect_to(): Slot {dest_name} already has {dest.max_incoming_connections} incoming connection(s) ({self.get_local_circuit_id()})")

        for parent in [source.parent, dest.parent]:
            if (parent is not self) and (parent.get_local_circuit_id() not in self.element_map.keys()):
                raise Exception(f"Circuit::connect_to(): Element {parent.get_local_circuit_id()} not found in Circuit (source:{source.parent.get_local_circuit_id()},dest:{dest.parent.get_local_circuit_id()}).")
            
        self.connection_map_reversed.setdefault(dest_name, [])
        self.connection_map_reversed[dest_name].append(source)

    def generate_kernel(self):
        self.compute_kernel = compute_kernel_factory(self.output_slot_map)

    def define_circuit_structure(self):
        # --- circuit description ---
        self.generate_kernel()

    def register_output_slot(self, slot_id : str, max_incoming_connections : int = 42) -> Slot:
        """Register output slot. For a circuit output slots receive input from internal state which are passed along as output."""
        slot = super().register_output_slot(slot_id)
        slot.max_incoming_connections = max_incoming_connections
        self.connection_map_reversed.setdefault(slot.get_local_circuit_id(), [])
        return slot
        
