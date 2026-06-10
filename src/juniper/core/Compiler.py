from __future__ import annotations

from .Circuit import Circuit
from .Element import Element
from .Slot import Slot
from .Buffer import Buffer
from .Step import Step
from .RuntimeInfo import CompileInfo
from .RuntimeInfo import ElementRef
from ..util import util_jax
from ..util import util



class Compiler:
    """Turn a user-defined Circuit into runtime metadata.

    The Compiler is the only place that walks the graph for compilation:
    it infers slot/buffer specs, validates special contracts, and builds the
    CompileInfo object consumed by Engine.

    Pseudocode:
        compile(top)
            compile_circuit(top)
                compile_circuit(child)
                    local_compile_info[child] = child_info
                local_compile_info[top] = flattened_top_info
            compile_info = local_compile_info[top]
            return compile_info
    """

    def __init__(self):
        """Create an empty compiler instance for one compilation run."""
        self.circuit = None
        self.local_compile_info : dict[Circuit, CompileInfo] = {}
        self.compile_info : CompileInfo | None = None
        self.compiled_element_map : dict[Circuit, dict[tuple[str, ...], Element]] = {}

    def compile(self, circuit : Circuit):
        """Compile a top-level circuit and return only its runtime CompileInfo.
        """
        self.circuit = circuit
        circuit.generate_kernel()
        self.compile_circuit(circuit, {})
        try:
            assert circuit.is_compiled
        except:
            failed_elements = self.gather_uncompiled_elements(circuit=circuit)
            element_paths = [element_to_path_str(element) for element in failed_elements]
            
            raise Exception(f"The circuit '{circuit.get_name()}' could not be compiled. \nThese elements failed to compile: {element_paths}")

        self.compile_info = self.local_compile_info[self.circuit]
        return self.compile_info

    def compile_circuit(self, circuit : Circuit, input_slots : dict[str, list[Slot]]) -> bool:
        """Infer a circuit by repeatedly compiling elements whose inputs are known.

        Pseudocode:
            while something changed:
                infer this circuit's input slots
                for each child element:
                    compile nested circuit or step
                infer this circuit's public output slots
            cache this circuit's local CompileInfo
        """
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

            output_slot_updated = self.compile_circuit_output_slots(circuit)
            if output_slot_updated:
                state_updated = True

        self.check_circuit_compiled(circuit)
        self.refresh_compile_info(circuit)
        return state_updated

    def compile_element(self, element : Element, input_slots : dict[str, list[Slot]]) -> bool:
        """Compile element as either a nested circuit or a step."""
        if isinstance(element, Circuit):
            return self.compile_circuit(element, input_slots)
        if isinstance(element, Step):
            return self.compile_step(element, input_slots)
        raise NotImplementedError(f"No compile behavior specified for element ({element.get_name()})")


    def compile_step(self, step : Step, input_slots : dict[str, list[Slot]]) -> bool:
        """Infer step's input slots, output slots, and buffers."""
        input_compile_info_updated, input_specs = self.compile_input_slots(step, input_slots=input_slots)
        output_compile_info_updated = self.compile_step_output_slots(step, input_specs=input_specs)
        buffer_compile_info_updated = self.compile_buffers(step)

        state_updated = input_compile_info_updated | output_compile_info_updated | buffer_compile_info_updated
        self.check_step_compiled(step)
        return state_updated

    def compile_input_slots(self, element : Element, input_slots : dict[str, list[Slot]]):
        """Merge connected sources and write them into an element's inputs."""
        input_specs = {}
        state_updated = False
        for slot_id, slot in element.input_slot_map.items():
            if slot.is_compiled:
                continue
            shape, dtype = self.merge_slot_compile_info(element, input_slots.get(slot.get_name(), []))
            if shape is None:
                continue
            if slot.shape != shape or slot.dtype != dtype:
                slot.shape = shape
                slot.dtype = dtype
                self.check_slot_compiled(slot)
                state_updated = True
            input_specs[slot_id] = (shape, dtype)
        return state_updated, input_specs

    def compile_step_output_slots(self, step : Step, input_specs : dict):
        """Ask a step to infer output specs from already known input specs."""
        output_state_updated = False
        output_shapes = step.infer_output_shapes(input_specs)
        output_dtypes = step.infer_output_dtypes(input_specs)
        for slot_id, shape in output_shapes.items():
            if slot_id not in step.output_slot_map or shape is None:
                continue
            slot = step.output_slot_map[slot_id]
            if slot.is_compiled:
                continue
            dtype = output_dtypes.get(slot_id, util_jax.cfg["jdtype"])
            if slot.shape != shape or slot.dtype != dtype:
                slot.shape = shape
                slot.dtype = dtype
                self.check_slot_compiled(slot)
                output_state_updated = True
        return output_state_updated

    def compile_circuit_output_slots(self, circuit : Circuit):
        """Infer a circuit's public output slots from its internal connections."""
        output_slot_updated = False
        for slot in circuit.output_slot_map.values():
            slot_name = slot.get_name()
            sources = circuit.connection_map_reversed[slot_name]
            shape, dtype = self.merge_slot_compile_info(circuit, sources)
            if shape is None:
                continue
            if slot.shape != shape or slot.dtype != dtype:
                slot.shape = shape
                slot.dtype = dtype
                self.check_slot_compiled(slot)
                output_slot_updated = True
        return output_slot_updated

    def compile_buffers(self, step : Step):
        """Resolve a step's buffer specs after slot shapes are known."""
        buffer_updated = False
        for buffer_id, buffer in step.buffer_map.items():
            shape = buffer.shape
            dtype = util_jax.cfg["jdtype"] if buffer.dtype is None else buffer.dtype
            permanent = False if buffer.permanent is None else buffer.permanent

            if not buffer.is_compiled:
                buffer.shape = shape
                buffer.dtype = dtype
                buffer.permanent = permanent
                self.check_buffer_compiled(buffer)
                buffer_updated = True
        return buffer_updated

    def check_buffer_compiled(self, buffer : Buffer):
        """Check if a buffers shape, type and permanency are known and is therefore compiled."""
        if buffer.shape is not None and buffer.dtype is not None and buffer.permanent is not None:
            buffer.is_compiled = True
        return buffer.is_compiled

    def check_slot_compiled(self, slot : Slot):
        """Check if a Slot is compiled."""
        if slot.shape is not None and slot.dtype is not None:
            slot.is_compiled = True
        return slot.is_compiled

    def check_step_compiled(self, step : Step):
        """Mark a step compiled only when all required slots and buffers are known."""
        for buffer in step.buffer_map.values():
            self.check_buffer_compiled(buffer)
            if not buffer.is_compiled:
                step.is_compiled = False
                return False
        for slot in step.input_slot_map.values():
            self.check_slot_compiled(slot)
            if not slot.is_compiled and step.needs_input_connections and not step.is_source:
                step.is_compiled = False
                return False
        for slot in step.output_slot_map.values():
            self.check_slot_compiled(slot)
            if not slot.is_compiled:
                step.is_compiled = False
                return False

        step.is_compiled = True
        return True

    def check_circuit_compiled(self, circuit : Circuit):
        """Mark a circuit compiled only when elements and public slots are known."""
        for element in circuit.element_map.values():
            if element is not circuit and not element.is_compiled:
                circuit.is_compiled = False
                return False
        for slot in circuit.input_slot_map.values():
            self.check_slot_compiled(slot)
            if not slot.is_compiled:
                circuit.is_compiled = False
                return False
        for slot in circuit.output_slot_map.values():
            self.check_slot_compiled(slot)
            if not slot.is_compiled:
                circuit.is_compiled = False
                return False

        circuit.is_compiled = True
        return True

    def merge_slot_compile_info(self, element : Element, sources : list[Slot]):
        """Combine incoming source slot specs into one input shape/dtype."""
        known_sources = [source for source in sources if source.is_compiled]
        if len(known_sources) == 0:
            return None, None

        shape = known_sources[0].shape
        dtype = known_sources[0].dtype or util_jax.cfg["jdtype"]
        for source in known_sources[1:]:
            source_shape = source.shape
            if source_shape != shape:
                if util._is_scalar_shape(shape):
                    shape = source_shape
                elif not util._is_scalar_shape(source_shape):
                    raise ValueError(f"Step {element.get_name()} received incompatible input shapes {shape} and {source_shape}")
            if source.dtype is not None:
                dtype = source.dtype
        return shape, dtype

    def empty_compile_info(self, circuit : Circuit) -> CompileInfo:
        """Create an empty CompileInfo container for one circuit."""
        return CompileInfo(
            circuit=circuit,
            compiled_elements={},
            dynamic=[],
            static=[],
            sources=[],
            sinks=[],
            sub_processes=[],
            kernel_map={},
        )

    def compile_entry(self, path : tuple[str, ...], element : Element) -> ElementRef:
        """Create the stable runtime reference for an element."""
        return ElementRef(path=path, element=element)

    def collect_compile_info(self, circuit : Circuit) -> CompileInfo:
        """Gather compiled elements, endpoints, and kernels."""
        compile_info = self.empty_compile_info(circuit)

        for element_name, element in circuit.element_map.items():
            if not element.is_compiled:
                continue

            element_path = (element_name,)
            element_ref = self.compile_entry(element_path, element)
            compile_info.compiled_elements[element_path] = element_ref
            if element.is_dynamic:
                compile_info.dynamic.append(element_ref)
            else:
                compile_info.static.append(element_ref)
            if element.is_source:
                compile_info.sources.append(element_ref)
            if element.is_sink:
                compile_info.sinks.append(element_ref)
            if element.manages_sup_process:
               compile_info.sub_processes.append(element_ref)

            if isinstance(element, Circuit):
                child_info = self.local_compile_info[element]
                for child_path, child_ref in child_info.compiled_elements.items():
                    path = (element_name,) + child_path
                    compile_info.compiled_elements[path] = self.compile_entry(path, child_ref.element)
                for child_ref in child_info.dynamic:
                    compile_info.dynamic.append(self.compile_entry((element_name,) + child_ref.path, child_ref.element))
                for child_ref in child_info.static:
                    compile_info.static.append(self.compile_entry((element_name,) + child_ref.path, child_ref.element))
                for child_ref in child_info.sources:
                    compile_info.sources.append(self.compile_entry((element_name,) + child_ref.path, child_ref.element))
                for child_ref in child_info.sinks:
                    compile_info.sinks.append(self.compile_entry((element_name,) + child_ref.path, child_ref.element))
                for child_ref in child_info.sub_processes:
                    compile_info.sub_processes.append(self.compile_entry((element_name,) + child_ref.path, child_ref.element))
                compile_info.kernel_map[element_name] = {
                    "kernel": element.compute_kernel,
                    "sub_kernel": child_info.kernel_map,
                }
            else:
                compile_info.kernel_map[element_name] = {
                    "kernel": element.compute_kernel,
                    "sub_kernel": None,
                }

        return compile_info

    def refresh_compile_info(self, circuit : Circuit):
        """Rebuild and cache CompileInfo for a circuit after inference changes."""
        self.local_compile_info[circuit] = self.collect_compile_info(circuit)
        return self.local_compile_info[circuit]

    def gather_uncompiled_elements(self, circuit : Circuit) -> list[Element]:
        uncompiled_elements = []
        for element in circuit.element_map.values():
            if not element.is_compiled:
                uncompiled_elements.append(element)
            if isinstance(element, Circuit):
                uncompiled_elements += self.gather_uncompiled_elements(element)
        return uncompiled_elements
    

def element_to_path_str(element : Element):
    """converts the element object into a path string indicating its position in the architecture-"""
    path = element.get_path()
    path_str=""
    for parent in path:
        path_str += parent.get_name() + "."
    path_str = path_str[:-1]
    return path_str