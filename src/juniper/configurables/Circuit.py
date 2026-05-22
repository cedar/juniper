from __future__ import annotations
from typing import Callable
from typing import Optional

from .Slot import Slot
from .Element import Element
import jax.numpy as jnp

def compute_kernel_factory(output_slot_map : dict[str,Slot], input_slot_map : dict[str,Slot], connection_map_reversed: dict[str,list[Slot]]) -> Callable[[dict, dict, Optional[dict]], dict]:
    def compute_kernel(input : dict, inner_state : dict, **kwargs : Optional[dict]) -> dict:
        new_state = inner_state.copy()

        # gather sub-inputs of sub-circuits
        sub_inputs = {}
        for dest, sources in connection_map_reversed.items():
            dest_name, dest_slot_id = dest.split(".")
            sub_inputs[dest_name] = {dest_slot_id: []}
            for source in sources:
                source_name, source_slot_id = source.get_name().split(".")
                sub_inputs[dest_name][dest_slot_id].append(inner_state[source_name][source_slot_id])

        # gather external input sums for sub-inputs
        for slot_name, dest in input_slot_map.items():
            dest_name, dest_slot_id = dest.get_name().split(".")
            if dest.get_name() in connection_map_reversed.keys():
                sub_inputs[dest_name][dest_slot_id].append(jnp.sum(input[slot_name], axis=0))
            else:
                sub_inputs[dest_name] = {dest_slot_id: [jnp.sum(input[slot_name], axis=0)]}

        # compute sub-circuits
        for element in connection_map_reversed.keys():
            element_name, _ = element.split(".")
            kernel = kwargs["kernel_map"][element_name]["kernel"]
            sub_kernel = kwargs["kernel_map"][element_name]["sub_kernel"]
            prng_keys = kwargs["prng_keys"][element_name]
            new_state[element_name] =  kernel(sub_inputs, inner_state[element_name], **{"prng_keys": prng_keys, "kernel_map":sub_kernel})

        # set output
        out = {}
        for slot_name, out_element in output_slot_map.items():
            out_element_name, out_element_slot_id = out_element.get_name().split(".")
            out[slot_name] = new_state[out_element_name][out_element_slot_id]

        out["inner_state"] = new_state
        return out # {"inner_state": {"step1":{"buffer":123, "out":123}, "step2"...}, "out_slot1":123, "out_slot2":123,...}

    return compute_kernel

class Circuit(Element):
    _current : Circuit | None = None

    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params)
        self.element_map : dict[str,Element] = {}
        self.connection_map_reversed : dict[str, list[Slot]] = {}
        self.compute_kernel : Callable[[dict, dict, Optional[dict]], dict] = None

        with self:
            self.build_circuit()
        
        self.parent_circuit.add_element(self)

    @classmethod
    def parent_circuit(cls : Circuit) -> Circuit | None:
        if cls._current is None:
            raise RuntimeError("No active circuit. This should never happen.")
        return cls._current
    
    def __enter__(self) -> Circuit:
        self._previous_circuit = Circuit._current
        Circuit._current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Circuit._current = self._previous_circuit
        return False

    def add_element(self, element : Element):
        element_name = element.get_name()
        if element_name in self.element_map.keys():
            raise Exception(f"Circuit::add_element(): Element {element_name} already exists in Circuit ({self.get_name()})")
        self.element_map[element_name] = element
        for slot in element.input_slot_map.values():
            self.connection_map_reversed[slot.get_name()] = []
    
    def get_elements(self) -> dict[str,Element]:
        return self.element_map
    
    def get_element(self, name : str) -> Element:
        if name not in self.element_map:
            raise Exception(f"Architecture::get_element(): Element {name} not found in Architecture")
        return self.element_map[name]
    
    def get_incoming_elements(self, dest : Element) -> list[Slot]:
        incoming_steps = []
        if isinstance(dest, Slot):
            incoming_steps = self.connection_map_reversed[dest.get_name()]
        else:
            for slot in dest.input_slot_map.values():
                incoming_steps += self.connection_map_reversed[slot.get_name()]
        return incoming_steps
    
    def connect_to(self, source : Slot, dest : Slot):
        source_name = source.get_name()
        dest_name = dest.get_name()
        
        for name in [source._parent.get_name(), dest._parent.get_name()]:
            if name not in self.element_map.keys():
                raise Exception(f"Circuit::connect_to(): Element {name} not found in Circuit ({self.get_name()})")
            
        if source_name in self.connection_map_reversed[dest_name]:
            raise Exception(f"Circuit::connect_to(): Connection from {source_name} to {dest_name} already exists ({self.get_name()})")
        
        if len(self.connection_map_reversed[dest_name]) >= dest.max_incoming_connections:
            raise Exception(f"Circuit::connect_to(): Slot {dest_name} already has {dest.max_incoming_connections} incoming connection(s) ({self.get_name()})")

        self.connection_map_reversed[dest_name].append(source)
    
    def set_input(self, input_slot_id : str, dest_slot : Slot, max_incoming_connections : int = 1):
        self.register_input_slot(slot_id=input_slot_id, max_incoming_connections=max_incoming_connections)
        # Register internal slot connection
        self.input_slot_map[input_slot_id] = dest_slot

    def set_output(self, output_slot_id : str, dest_slot : Slot):
        self.register_output_slot(slot_id= output_slot_id)
        # Register internal slot connection
        self.output_slot_map[output_slot_id] = dest_slot

    def generate_kernel(self):
        self.compute_kernel = compute_kernel_factory(self.output_slot_map, self.input_slot_map, self.connection_map_reversed)

    def build_circuit(self):
        pass
