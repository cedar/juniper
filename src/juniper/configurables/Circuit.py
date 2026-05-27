from __future__ import annotations
from typing import Callable
from typing import Optional

from .Slot import Slot
from .Element import Element
import jax.numpy as jnp

def compute_kernel_factory(element_map : dict[str,Element], output_slot_map : dict[str,Slot], input_slot_map : dict[str,Slot], connection_map_reversed: dict[str,list[Slot]]) -> Callable[[dict, dict, Optional[dict]], dict]:
    def compute_kernel(input : dict, inner_state : dict, **kwargs : Optional[dict]) -> dict:
        new_state = inner_state.copy()

        for element_name, element in element_map.items():
            sub_inputs = {}
            for input_slot_id, input_slot in element.input_slot_map.items():
                input_values = []
                for source in connection_map_reversed[input_slot.get_name()]:
                    source_name, source_slot_id = source.get_name().split(".")
                    if source_name == source.parent.get_name():
                        if source.parent is element:
                            raise Exception(f"Circuit::compute_kernel(): Element {element_name} cannot receive input from itself")
                        if source.parent in element_map.values():
                            input_values.append(new_state[source_name][source_slot_id])
                        else:
                            input_values.append(input[source_slot_id])

                if len(input_values) == 0:
                    continue

                input_sum = input_values[0]
                for value in input_values[1:]:
                    input_sum = input_sum + value
                sub_inputs[input_slot_id] = input_sum

            kernel = kwargs["kernel_map"][element_name]["kernel"]
            sub_kernel = kwargs["kernel_map"][element_name]["sub_kernel"]
            prng_keys = kwargs["prng_keys"][element_name]
            new_state[element_name] =  kernel(sub_inputs, inner_state[element_name], **{"prng_key": prng_keys, "prng_keys": prng_keys, "kernel_map":sub_kernel})

        # set output
        out = new_state.copy()
        for slot_name, out_slot in output_slot_map.items():
            incoming = connection_map_reversed[out_slot.get_name()]
            if len(incoming) == 0:
                continue
            source = incoming[0]
            out_element_name, out_element_slot_id = source.get_name().split(".")
            if source.parent in element_map.values():
                out[slot_name] = new_state[out_element_name][out_element_slot_id]
            else:
                out[slot_name] = input[out_element_slot_id]

        out["inner_state"] = new_state
        return out # {"inner_state": {"step1":{"buffer":123, "out":123}, "step2"...}, "out_slot1":123, "out_slot2":123,...}

    return compute_kernel

class Circuit(Element):
    _current : Circuit | None = None

    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params)
        self.element_map : dict[str,Element] = {}
        self.connection_map_reversed : dict[str, list[Slot]] = {}
        self.compile_info : dict = self.empty_compile_info()

        with self:
            self.build_circuit()
        
        self.parent.add_element(self)

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
        
        for parent in [source.parent, dest.parent]:
            if parent is not self and parent.get_name() not in self.element_map.keys():
                raise Exception(f"Circuit::connect_to(): Element {parent.get_name()} not found in Circuit (source:{source.parent.get_name()},dest:{dest.parent.get_name()}). This should never happen.")
            
        self.connection_map_reversed.setdefault(dest_name, [])

        if source in self.connection_map_reversed[dest_name]:
            raise Exception(f"Circuit::connect_to(): Connection from {source_name} to {dest_name} already exists ({self.get_name()})")
        
        if len(self.connection_map_reversed[dest_name]) >= dest.max_incoming_connections:
            raise Exception(f"Circuit::connect_to(): Slot {dest_name} already has {dest.max_incoming_connections} incoming connection(s) ({self.get_name()})")

        self.connection_map_reversed[dest_name].append(source)
    
    def set_input(self, input_slot_id : str, dest_slot : Slot, max_incoming_connections : int = 1):
        self.register_input_slot(slot_id=input_slot_id, max_incoming_connections=max_incoming_connections)
        # Register internal slot connection
        self.connect_to(self.input_slot_map[input_slot_id], dest_slot)

    def set_output(self, output_slot_id : str, source_slot : Slot):
        self.register_output_slot(slot_id= output_slot_id)
        # Register internal slot connection
        self.connect_to(source_slot, self.output_slot_map[output_slot_id])

    def generate_kernel(self):
        self.compute_kernel = compute_kernel_factory(self.element_map, self.output_slot_map, self.input_slot_map, self.connection_map_reversed)

    def build_circuit(self):
        # --- circuit description ---
        self.generate_kernel()

    def empty_compile_info(self) -> dict:
        return {
            "circuit": self,
            "known_state": {},
            "dynamic": [],
            "static": [],
            "sources": [],
            "sinks": [],
            "sub_processes": [],
            "state_info": {},
            "kernel_map": {},
            "children": {},
        }

    def _prefixed_path(self, prefix : str, path : tuple[str, ...]) -> tuple[str, ...]:
        return (prefix,) + path

    def _compile_entry(self, path : tuple[str, ...], element : Element) -> dict:
        return {"path": path, "element": element}

    def _element_state_info(self, element : Element) -> dict:
        return {
            "element": element,
            "kind": "dynamic" if element.is_dynamic else "static",
            "is_dynamic": element.is_dynamic,
            "is_source": element.is_source,
            "is_sink": element.is_sink,
            "manages_sup_process": element.manages_sup_process,
            "input_slots": element.input_slot_map,
            "output_slots": element.output_slot_map,
            "buffer_map": getattr(element, "buffer_map", {}),
        }

    def collect_compile_info(self) -> dict:
        compile_info = self.empty_compile_info()

        for element_name, element in self.element_map.items():
            if not element.is_compiled:
                continue

            element_path = (element_name,)
            compile_info["known_state"][element_path] = element
            if element.is_dynamic:
                compile_info["dynamic"].append(self._compile_entry(element_path, element))
            else:
                compile_info["static"].append(self._compile_entry(element_path, element))
            if element.is_source:
                compile_info["sources"].append(self._compile_entry(element_path, element))
            if element.is_sink:
                compile_info["sinks"].append(self._compile_entry(element_path, element))
            if element.manages_sup_process:
                compile_info["sub_processes"].append(self._compile_entry(element_path, element))

            element_info = self._element_state_info(element)
            compile_info["state_info"][element_path] = element_info

            if isinstance(element, Circuit):
                child_info = element.compile_info
                compile_info["children"][element_name] = child_info
                compile_info["state_info"][element_path]["children"] = child_info["state_info"]
                for child_path, child_element in child_info["known_state"].items():
                    compile_info["known_state"][self._prefixed_path(element_name, child_path)] = child_element
                for key in ["dynamic", "static", "sources", "sinks", "sub_processes"]:
                    for entry in child_info[key]:
                        compile_info[key].append({
                            "path": self._prefixed_path(element_name, entry["path"]),
                            "element": entry["element"],
                        })
                compile_info["kernel_map"][element_name] = {
                    "kernel": element.compute_kernel,
                    "sub_kernel": child_info["kernel_map"],
                }
            else:
                compile_info["kernel_map"][element_name] = {
                    "kernel": element.compute_kernel,
                    "sub_kernel": None,
                }

        return compile_info

    def refresh_compile_info(self):
        self.compile_info = self.collect_compile_info()
        return self.compile_info

    def compile_state(self, input_slots : dict[str,Slot]):
        state_updated = False
        sub_state_updated = True
        
        while sub_state_updated:
            # update sub-elements
            sub_state_updated = False
            for element_name, element in self.element_map.items():
                if (element_name,) in self.compile_info["known_state"]:
                    continue
                input_slots = {slot.get_name(): self.connection_map_reversed[slot.get_name()] for slot in element.input_slot_map.values()}
                sub_state_updated = element.compile_state(input_slots)

                # if any sub-state updated, the parent state also updated
                if sub_state_updated:
                    state_updated = True

                # if element is compiled successfully, store its meta data in state_info dicts
                if element.is_compiled and sub_state_updated:
                    element_path = (element_name,)
                    self.compile_info["known_state"][element_path] = element

                    if element.is_dynamic:
                        self.compile_info["dynamic"].append(self._compile_entry(element_path, element))
                    else:
                        self.compile_info["static"].append(self._compile_entry(element_path, element))

                    if element.is_sink:
                        self.compile_info["sinks"].append(self._compile_entry(element_path, element))
                    elif element.is_source:
                        self.compile_info["sources"].append(self._compile_entry(element_path, element))

                    if element.manages_sup_process:
                        self.compile_info["sub_processes"].append(self._compile_entry(element_path, element))
            
            # update input and output slots
            for input_slot in self.input_slot_map.values():
                continue
            for output_slot in self.output_slot_map.values():
                continue

        self.check_compiled()
        self.refresh_compile_info()

        return state_updated
    
    def check_compiled(self):
        for element in self.element_map.values():
            if not element.is_compiled:
                self.is_compiled = False
                return False
        for slot in self.input_slot_map.values():
            if not slot.is_compiled:
                self.is_compiled=False
                return False
        for slot in self.output_slot_map.values():
            if not slot.is_compiled:
                self.is_compiled=False
                return False

        # all sub-elements and slots are comiled -> circuit is compiled
        self.is_compiled = True
        return self.is_compiled       
        
