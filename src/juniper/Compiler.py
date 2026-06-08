from __future__ import annotations

from .configurables.Circuit import Circuit
from .configurables.Element import Element
from .configurables.Slot import Slot
from .configurables.Step import Step


class Compiler:
    def __init__(self):
        self.circuit = None
        self.compile_info_map : dict[Circuit, dict] = {}
        self.compiled_element_map : dict[Circuit, dict[tuple[str, ...], Element]] = {}

    def compile(self, circuit : Circuit):
        self.circuit = circuit
        circuit.generate_kernel()
        self.compile_circuit(circuit, {})
        return self.compile_info_map[self.circuit]

    def compile_element(self, element : Element, input_slots : dict[str, list[Slot]]) -> bool:
        if isinstance(element, Circuit):
            return self.compile_circuit(element, input_slots)
        if isinstance(element, Step):
            return self.compile_step(element, input_slots)
        raise NotImplementedError(f"No compile behavior specified for element ({element.get_name()})")

    def compile_circuit(self, circuit : Circuit, input_slots : dict[str, list[Slot]]) -> bool:
        state_updated = False
        sub_state_updated = True
        self.compiled_element_map[circuit] = {}

        while sub_state_updated:
            input_slots_updated, _ = self.compile_input_slots(circuit, input_slots=input_slots)
            if input_slots_updated:
                state_updated = True

            sub_state_updated = False
            for element_name, element in circuit.element_map.items():
                if (element_name,) in self.compiled_element_map[circuit]:
                    continue

                element_input_slots = {
                    slot.get_name(): circuit.connection_map_reversed[slot.get_name()]
                    for slot in element.input_slot_map.values()
                }
                element_state_updated = self.compile_element(element, element_input_slots)

                if element_state_updated:
                    state_updated = True
                    sub_state_updated = True

                if element.is_compiled:
                    self.compiled_element_map[circuit][(element_name,)] = element

            output_slot_updated, _ = self.compile_circuit_output_slots(circuit)
            if output_slot_updated:
                state_updated = True

        self.check_circuit_compiled(circuit)
        self.refresh_compile_info(circuit)
        return state_updated

    def compile_step(self, step : Step, input_slots : dict[str, list[Slot]]) -> bool:
        input_compile_info_updated, input_specs = self.compile_input_slots(step, input_slots=input_slots)
        output_compile_info_updated = self.compile_step_output_slots(step, input_specs=input_specs)
        buffer_compile_info_updated = self.compile_buffers(step)

        state_updated = input_compile_info_updated | output_compile_info_updated | buffer_compile_info_updated
        self.check_step_compiled(step)
        return state_updated

    def compile_input_slots(self, element : Element, input_slots : dict[str, list[Slot]]):
        input_specs = {}
        state_updated = False
        for slot_id, slot in element.input_slot_map.items():
            shape, dtype = self.merge_slot_compile_info(element, input_slots.get(slot.get_name(), []))
            if shape is None:
                continue
            if slot.shape != shape or slot.dtype != dtype:
                slot.shape = shape
                slot.dtype = dtype
                slot.check_compiled()
                state_updated = True
            input_specs[slot_id] = (shape, dtype)
        return state_updated, input_specs

    def compile_step_output_slots(self, step : Step, input_specs : dict):
        output_state_updated = False
        output_shapes = step.infer_output_shapes(input_specs)
        output_dtypes = step.infer_output_dtypes(input_specs)
        for slot_id, shape in output_shapes.items():
            if slot_id not in step.output_slot_map or shape is None:
                continue
            slot = step.output_slot_map[slot_id]
            dtype = output_dtypes.get(slot_id, step._default_dtype())
            if slot.shape != shape or slot.dtype != dtype:
                slot.shape = shape
                slot.dtype = dtype
                slot.check_compiled()
                output_state_updated = True
        return output_state_updated

    def compile_circuit_output_slots(self, circuit : Circuit):
        output_slot_updated = False
        output_specs = {}
        for slot in circuit.output_slot_map.values():
            slot_name = slot.get_name()
            sources = circuit.connection_map_reversed[slot_name]
            shape, dtype = self.merge_slot_compile_info(circuit, sources)
            if shape is None:
                continue
            if slot.shape != shape or slot.dtype != dtype:
                slot.shape = shape
                slot.dtype = dtype
                slot.check_compiled()
                output_slot_updated = True
            output_specs[slot.get_slot_id()] = (shape, dtype)
        return output_slot_updated, output_specs

    def compile_buffers(self, step : Step):
        buffer_updated = False
        for buffer_id, buffer in step.buffer_map.items():
            shape = self.resolve_shape(step, buffer.shape)
            if shape is None and buffer_id in step.output_slot_map:
                shape = step.output_slot_map[buffer_id].shape
            if shape is None:
                continue
            if buffer.dtype is not None:
                dtype = buffer.dtype
            elif buffer_id in step.output_slot_map:
                dtype = step.output_slot_map[buffer_id].dtype or step._default_dtype()
            else:
                dtype = step._default_dtype()
            if buffer.shape != shape or buffer.dtype != dtype:
                buffer.shape = shape
                buffer.dtype = dtype
                buffer.check_compiled()
                buffer_updated = True
        return buffer_updated

    def check_step_compiled(self, step : Step):
        for buffer in step.buffer_map.values():
            buffer.check_compiled()
            if not buffer.is_compiled:
                step.is_compiled = False
                return False
        for slot in step.input_slot_map.values():
            slot.check_compiled()
            if not slot.is_compiled and step.needs_input_connections and not step.is_source:
                step.is_compiled = False
                return False
        for slot in step.output_slot_map.values():
            slot.check_compiled()
            if not slot.is_compiled:
                step.is_compiled = False
                return False

        step.is_compiled = True
        return True

    def check_circuit_compiled(self, circuit : Circuit):
        for element in circuit.element_map.values():
            if element is not circuit and not element.is_compiled:
                circuit.is_compiled = False
                return False
        for slot in circuit.input_slot_map.values():
            slot.check_compiled()
            if not slot.is_compiled:
                circuit.is_compiled = False
                return False
        for slot in circuit.output_slot_map.values():
            slot.check_compiled()
            if not slot.is_compiled:
                circuit.is_compiled = False
                return False

        circuit.is_compiled = True
        return True

    def merge_slot_compile_info(self, element : Element, sources : list[Slot]):
        known_sources = [source for source in sources if source.check_compiled()]
        if len(known_sources) == 0:
            return None, None

        shape = known_sources[0].shape
        dtype = known_sources[0].dtype or element._default_dtype()
        for source in known_sources[1:]:
            source_shape = source.shape
            if source_shape != shape:
                if element._is_scalar_shape(shape):
                    shape = source_shape
                elif not element._is_scalar_shape(source_shape):
                    raise ValueError(f"Step {element.get_name()} received incompatible input shapes {shape} and {source_shape}")
            if source.dtype is not None:
                dtype = source.dtype
        return shape, dtype

    def resolve_shape(self, step : Step, shape):
        if isinstance(shape, str):
            if shape in step._params:
                value = step._params[shape]
                if isinstance(value, int):
                    return (value,)
                return tuple(value)
            return None
        if isinstance(shape, int):
            return (shape,)
        if shape is None:
            return None
        return tuple(shape)

    def empty_compile_info(self, circuit : Circuit) -> dict:
        return {
            "circuit": circuit,
            "compiled_elements": {},
            "dynamic": [],
            "static": [],
            "sources": [],
            "sinks": [],
            "sub_processes": [],
            "state_info": {},
            "kernel_map": {},
            "children": {},
        }

    def compile_entry(self, path : tuple[str, ...], element : Element) -> dict:
        return {"path": path, "element": element}

    def element_state_info(self, element : Element) -> dict:
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

    def collect_compile_info(self, circuit : Circuit) -> dict:
        compile_info = self.empty_compile_info(circuit)

        for element_name, element in circuit.element_map.items():
            if not element.is_compiled:
                continue

            element_path = (element_name,)
            compile_info["compiled_elements"][element_path] = element
            if element.is_dynamic:
                compile_info["dynamic"].append(self.compile_entry(element_path, element))
            else:
                compile_info["static"].append(self.compile_entry(element_path, element))
            if element.is_source:
                compile_info["sources"].append(self.compile_entry(element_path, element))
            if element.is_sink:
                compile_info["sinks"].append(self.compile_entry(element_path, element))
            if element.manages_sup_process:
                compile_info["sub_processes"].append(self.compile_entry(element_path, element))

            compile_info["state_info"][element_path] = self.element_state_info(element)

            if isinstance(element, Circuit):
                child_info = self.compile_info_map[element]
                compile_info["children"][element_name] = child_info
                compile_info["state_info"][element_path]["children"] = child_info["state_info"]
                for child_path, child_element in child_info["compiled_elements"].items():
                    compile_info["compiled_elements"][(element_name,) + child_path] = child_element
                for key in ["dynamic", "static", "sources", "sinks", "sub_processes"]:
                    for entry in child_info[key]:
                        compile_info[key].append({
                            "path": (element_name,) + entry["path"],
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

    def refresh_compile_info(self, circuit : Circuit):
        self.compile_info_map[circuit] = self.collect_compile_info(circuit)
        return self.compile_info_map[circuit]
