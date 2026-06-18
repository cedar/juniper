from __future__ import annotations
from typing import Callable
from typing import Optional

from ..backend.Exceptions import CircuitError
from ..backend.Exceptions import CircuitConnectionError

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
        for element_name, element in self.element_map.items():
            if getattr(self, element_name, None) is element:
                delattr(self, element_name)
        for slot_id, slot in self.input_slot_map.items():
            if getattr(self, slot_id, None) is slot:
                delattr(self, slot_id)
            if slot.get_local_circuit_id() in self.parent.connection_map_reversed.keys():
                self.parent.connection_map_reversed.pop(slot.get_local_circuit_id())
        for slot_id, slot in self.output_slot_map.items():
            if getattr(self, slot_id, None) is slot:
                delattr(self, slot_id)
            if slot.get_local_circuit_id() in self.parent.connection_map_reversed.keys():
                self.parent.connection_map_reversed.pop(slot.get_local_circuit_id())

        if self.get_local_circuit_id() in self.parent.element_map.keys():
            self.parent.element_map.pop(self.get_local_circuit_id())

        self.element_map = {}
        self.input_slot_map = {}
        self.output_slot_map = {}
        self.connection_map_reversed = {}
        self.is_compiled = False
        self.compute_kernel = None


    def add_element(self, element : Element):
        element_name = element.get_local_circuit_id()
        if element_name in self.element_map.keys():
            raise CircuitError(f"Circuit::add_element(): Element {element_name} already exists in circuit {self.get_path_str()}")
        if self is element:
            raise CircuitError(f"Circuit::add_element(): A circuit can't be added as a sub-element to itself ({self.get_path_str()}).")
        self.element_map[element_name] = element
        for slot in element.input_slot_map.values():
            #print(slot.get_local_circuit_id())
            self.connection_map_reversed[slot.get_local_circuit_id()] = []

        if (element.get_local_circuit_id() in self.input_slot_map.keys()) or (element.get_local_circuit_id() in self.input_slot_map.keys()):
            raise CircuitError(f"A sub-element of a circuit cannot have the same name as its input or output slot ({element.get_path_str()})")
        elif getattr(self, element.get_local_circuit_id(), None) is not None:
            raise CircuitError(f"{element.self.get_local_circuit_id()} is already registed in circuit {element.get_path_str()}")
        else:
            setattr(self, element.get_local_circuit_id(), element)
    
    def get_elements(self) -> dict[str,Element]:
        return self.element_map
    
    def get_element(self, name : str) -> Element:
        if name not in self.element_map:
            raise CircuitError(f"Circuit::get_element(): Element {name} not found in circuit {self.get_path_str()}")
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
            raise CircuitConnectionError(f"Circuit::connect_to(): Connection from {source_name} to {dest_name} already exists in {self.get_path_str()}")
        
        if len(self.connection_map_reversed[dest_name]) >= dest.max_incoming_connections:
            raise CircuitConnectionError(f"Circuit::connect_to(): Slot {dest_name} already has {dest.max_incoming_connections} incoming connection(s) {self.get_path_str()}")

        for parent in [source.parent, dest.parent]:
            if (parent is not self) and (parent.get_local_circuit_id() not in self.element_map.keys()):
                raise CircuitConnectionError(f"Circuit::connect_to(): Element {parent.get_local_circuit_id()} not found in circuit {self.get_path_str()} (source:{source.parent.get_local_circuit_id()},dest:{dest.parent.get_local_circuit_id()})")
            
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
        
