from __future__ import annotations
from typing import Callable
from typing import Optional

from .Slot import Slot
from .Element import Element
from . import CircuitContext

def compute_kernel_factory(element_map : dict[str,Element], output_slot_map : dict[str,Slot], input_slot_map : dict[str,Slot], connection_map_reversed: dict[str,list[Slot]]) -> Callable[[dict, dict, Optional[dict]], dict]:
    def compute_kernel(input : dict, state : dict, **kwargs : Optional[dict]) -> dict:
        """
            - input: {"in0":jax_array, "in1":jax_array, ...}
            - state: {"element1": {"buffer":jax_array, "out0":jax_array,...}, "sub_circuit":{sub_state}, "out0":jax_array, "out1":...}
            - prng_keys: {"element1": key1, "sub_circtuit": {"sub_keys}, ...}
            - kernel_map: {"element1": {"kernel":kernel1, "sub_kernel":None}, "sub_circuit": {"kernel":kernel2, "sub_kernel": {sub_kernels}}}
        """

        new_state = state.copy()

        for element_name, element in element_map.items():
            sub_inputs = {}
            for input_slot_id, input_slot in element.input_slot_map.items():
                sub_inputs[input_slot_id] = 0
                input_values = []
                for source_slot in connection_map_reversed[input_slot.get_name()]:
                    source_name, source_slot_id = source_slot.get_name().split(".")
                    if source_slot.parent in element_map.values():
                        input_values.append(new_state[source_name][source_slot_id])
                    else:
                        input_values.append(input[source_slot_id])

                if len(input_values) == 0:
                    continue

                input_reduction = input_values[0]
                for value in input_values[1:]:
                    if element.input_aggregation == "product":
                        input_reduction = input_reduction * value
                    elif element.input_aggregation == "sum":
                        input_reduction = input_reduction + value
                sub_inputs[input_slot_id] = input_reduction

            kernel = kwargs["kernel_map"][element_name]["kernel"]
            sub_kernel = kwargs["kernel_map"][element_name]["sub_kernel"]
            prng_keys = kwargs["prng_keys"][element_name]
            element_out = kernel(sub_inputs, state[element_name], **{"prng_key": prng_keys, "prng_keys": prng_keys, "kernel_map":sub_kernel})
            new_state[element_name] = element_out

        # set output
        for slot_id, out_slot in output_slot_map.items():
            source_slots = connection_map_reversed[out_slot.get_name()]
            new_state[slot_id] = new_state[slot_id] * 0
            for source_slot in source_slots:
                source = source_slot.parent
                if source in element_map.values():
                    new_state[slot_id] = new_state[slot_id] + new_state[source.get_name()][source_slot.get_slot_id()]
                else:
                    new_state[slot_id] = new_state[slot_id] + input[source_slot.get_slot_id()]

        return new_state

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
        if self.get_name() in self.parent.element_map.keys():
            self.parent.element_map.pop(self.get_name())
        for slot in self.input_slot_map.values():
            if slot.get_name() in self.parent.connection_map_reversed.keys():
                self.parent.connection_map_reversed.pop(slot.get_name())
        for slot in self.output_slot_map.values():
            if slot.get_name() in self.parent.connection_map_reversed.keys():
                self.parent.connection_map_reversed.pop(slot.get_name())

        self.is_compiled = False
        self.compute_kernel = None


    def add_element(self, element : Element):
        element_name = element.get_name()
        if element_name in self.element_map.keys():
            raise Exception(f"Circuit::add_element(): Element {element_name} already exists in Circuit ({self.get_name()})")
        if self is element:
            raise Exception(f"Circuit::add_element(): A Circuit can't be added as a sub-element to itself ({self.get_name()}).")
        self.element_map[element_name] = element
        for slot in element.input_slot_map.values():
            #print(slot.get_name())
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
        if not isinstance(source, Slot) or not isinstance(dest, Slot):
            source = self.get_slot_from_connectable(source)
            dest = self.get_slot_from_connectable(dest)
        source_name = source.get_name()
        dest_name = dest.get_name()
        
        for parent in [source.parent, dest.parent]:
            if parent.get_name() not in self.element_map.keys():
                raise Exception(f"Circuit::connect_to(): Element {parent.get_name()} not found in Circuit (source:{source.parent.get_name()},dest:{dest.parent.get_name()}).")
            
        self.connection_map_reversed.setdefault(dest_name, [])

        if source in self.connection_map_reversed[dest_name]:
            raise Exception(f"Circuit::connect_to(): Connection from {source_name} to {dest_name} already exists ({self.get_name()})")
        
        max_incoming_connections = getattr(dest.parent, "_max_incoming_connections", {}).get(dest.get_slot_id(), dest.max_incoming_connections)
        if len(self.connection_map_reversed[dest_name]) >= max_incoming_connections:
            raise Exception(f"Circuit::connect_to(): Slot {dest_name} already has {max_incoming_connections} incoming connection(s) ({self.get_name()})")

        self.connection_map_reversed[dest_name].append(source)
    
    def set_input(self, input_slot_id : str, dest_slot, max_incoming_connections : int = 1):
        dest_slot = self.get_slot_from_connectable(dest_slot, dir="in")
        self.register_input_slot(slot_id=input_slot_id, max_incoming_connections=max_incoming_connections)
        # Register internal slot connection

        self.connection_map_reversed[dest_slot.get_name()].append(self.get_input_slot(input_slot_id))

    def set_output(self, output_slot_id : str, source_slot):
        source_slot = self.get_slot_from_connectable(source_slot, dir="out")
        self.register_output_slot(slot_id=output_slot_id)
        # Register internal slot connection

        out_slot_name  = self.get_output_slot(output_slot_id).get_name()
        if out_slot_name not in self.connection_map_reversed.keys():
            self.connection_map_reversed[out_slot_name] = [source_slot] 
        else:
            self.connection_map_reversed[out_slot_name].append(source_slot)
        

    def generate_kernel(self):
        self.compute_kernel = compute_kernel_factory(self.element_map, self.output_slot_map, self.input_slot_map, self.connection_map_reversed)

    def define_circuit_structure(self):
        # --- circuit description ---
        self.generate_kernel()
        
