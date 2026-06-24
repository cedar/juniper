from __future__ import annotations
import logging
from typing import Any
from .DataClasses import CompileInfo
from .DataClasses import ElementRef
from .Exceptions import CompilerError
from .Exceptions import ShapeInferenceError
from .Warnings import TypeInferenceWarning
import warnings

from ..frontend.Circuit import Circuit
from ..frontend.Element import Element
from ..frontend.Slot import Slot
from ..frontend.Buffer import Buffer
from ..frontend.Step import Step
from ...util import util_jax
from ...util import util




logger = logging.getLogger(__name__)
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

    @classmethod
    def compile(cls, circuit : Circuit):
        """Compile a top-level circuit and return only its runtime CompileInfo.
        """
        compiler = Compiler()
        compiler.circuit = circuit
        circuit.generate_kernel()
        compiler._compile_circuit(circuit, {})
        try:
            assert circuit.is_compiled
        except Exception as e:
            failed_elements = compiler._gather_uncompiled_elements(circuit=circuit)
            element_paths = [ElementRef(element).path_str for element in failed_elements]
            failure_traces, failure_sources = _trace_compile_failure_sources(failed_elements)
            print([failure_source.get_path_str() for failure_source in failure_sources])
            for trace in failure_traces:
                trace_str = ""
                for el in trace:
                    trace_str += el.get_path_str() + " -> "
                print(trace_str[:-3])
            raise CompilerError(f"The circuit '{circuit.get_local_circuit_id()}' could not be compiled. \nThese elements failed to compile: {element_paths}") from e

        compiler.compile_info = compiler.local_compile_info[compiler.circuit]
        return compiler.compile_info

    def _compile_circuit(self, circuit : Circuit, input_slots : dict[str, list[Slot]]) -> bool:
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
            input_slots_updated, _ = self._compile_input_slots(circuit, input_slots=input_slots)
            if input_slots_updated:
                state_updated = True

            sub_state_updated = False
            for element_name, element in circuit.element_map.items():

                element_input_slots = {
                    slot.get_local_circuit_id(): circuit.connection_map_reversed[slot.get_local_circuit_id()]
                    for slot in element.input_slot_map.values()
                }
                element_state_updated = self._compile_element(element, element_input_slots)

                if element_state_updated:
                    state_updated = True
                    sub_state_updated = True

                element_ref = ElementRef(element)
                if element.is_compiled and element_ref.path not in self.compiled_element_map[circuit]:
                    self.compiled_element_map[circuit][element_ref.path] = element

            output_slot_updated = self._compile_circuit_output_slots(circuit)
            if output_slot_updated:
                state_updated = True

        self._check_circuit_compiled(circuit)
        self._refresh_compile_info(circuit)
        return state_updated

    def _compile_element(self, element : Element, input_slots : dict[str, list[Slot]]) -> bool:
        """Compile element as either a nested circuit or a step."""
        if isinstance(element, Circuit):
            return self._compile_circuit(element, input_slots)
        if isinstance(element, Step):
            return self._compile_step(element, input_slots)
        raise NotImplementedError(f"No compile behavior specified for element ({element.get_local_circuit_id()})")

    def _compile_step(self, step : Step, input_slots : dict[str, list[Slot]]) -> bool:
        """Infer step's input slots, output slots, and buffers."""
        input_compile_info_updated, input_specs = self._compile_input_slots(step, input_slots=input_slots)
        output_compile_info_updated = self._compile_step_output_slots(step, input_specs=input_specs)
        buffer_compile_info_updated = self._compile_buffers(step)

        state_updated = input_compile_info_updated | output_compile_info_updated | buffer_compile_info_updated
        self._check_step_compiled(step)
        return state_updated

    def _compile_input_slots(self, element : Element, input_slots : dict[str, list[Slot]]):
        """Merge connected sources and write them into an element's inputs."""
        state_updated = False
        input_specs = {}
        for slot_id, slot in element.input_slot_map.items():
            shape, dtype = self._merge_slot_compile_info(element, input_slots.get(slot.get_local_circuit_id(), []))
            state_updated = _slot_changed(slot, shape, dtype)
            self._check_slot_compiled(slot)
            if slot.shape is not None:
                input_specs[slot_id] = (slot.shape, slot.dtype)
                
        return state_updated, input_specs

    def _compile_step_output_slots(self, step : Step, input_specs : dict):
        """Ask a step to infer output specs from already known input specs."""
        output_state_updated = False
        output_shapes = step.infer_output_shapes(input_specs)
        output_dtypes = step.infer_output_dtypes(input_specs)
        for slot_id, shape in output_shapes.items():
            slot = step.output_slot_map[slot_id]
            dtype = output_dtypes.get(slot_id, util_jax.cfg["jdtype"])
            output_state_updated = _slot_changed(slot, shape, dtype)
            self._check_slot_compiled(slot)
        return output_state_updated

    def _compile_circuit_output_slots(self, circuit : Circuit):
        """Infer a circuit's public output slots from its internal connections."""
        output_slot_updated = False
        for slot in circuit.output_slot_map.values():
            slot_name = slot.get_local_circuit_id()
            sources = circuit.connection_map_reversed[slot_name]
            shape, dtype = self._merge_slot_compile_info(circuit, sources)
            output_slot_updated = _slot_changed(slot, shape, dtype)
            self._check_slot_compiled(slot)
        return output_slot_updated

    def _compile_buffers(self, step : Step):
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
                self._check_buffer_compiled(buffer)
                buffer_updated = True
        return buffer_updated

    def _check_buffer_compiled(self, buffer : Buffer):
        """Check if a buffers shape, type and permanency are known and is therefore compiled."""
        if buffer.shape is not None and buffer.dtype is not None and buffer.permanent is not None:
            buffer.is_compiled = True
        return buffer.is_compiled

    def _check_slot_compiled(self, slot : Slot):
        """Check if a Slot is compiled."""
        if slot.shape is not None and slot.dtype is not None:
            slot.is_compiled = True
        return slot.is_compiled

    def _check_step_compiled(self, step : Step):
        """Mark a step compiled only when all required slots and buffers are known."""
        for buffer in step.buffer_map.values():
            self._check_buffer_compiled(buffer)
            if not buffer.is_compiled:
                step.is_compiled = False
                return False
        for slot in step.input_slot_map.values():
            self._check_slot_compiled(slot)
            if not slot.is_compiled and step.needs_input_connections and not step.is_source:
                step.is_compiled = False
                return False
        for slot in step.output_slot_map.values():
            self._check_slot_compiled(slot)
            if not slot.is_compiled:
                step.is_compiled = False
                return False

        step.is_compiled = True
        return True

    def _check_circuit_compiled(self, circuit : Circuit):
        """Mark a circuit compiled only when elements and public slots are known."""
        for element in circuit.element_map.values():
            if element is not circuit and not element.is_compiled:
                circuit.is_compiled = False
                return False
        for slot in circuit.input_slot_map.values():
            self._check_slot_compiled(slot)
            if not slot.is_compiled:
                circuit.is_compiled = False
                return False
        for slot in circuit.output_slot_map.values():
            self._check_slot_compiled(slot)
            if not slot.is_compiled:
                circuit.is_compiled = False
                return False

        circuit.is_compiled = True
        return True

    def _merge_slot_compile_info(self, element : Element, sources : list[Slot]):
        """Combine incoming source slot specs into one input shape/dtype."""
        known_sources = [source for source in sources if source.is_compiled]
        if len(known_sources) == 0:
            return None, None

        shape = known_sources[0].shape
        dtype = None
        raise_dtype_warning = False
        for source in known_sources[1:]:
            source_shape = source.shape
            if source_shape != shape:
                if util._is_scalar_shape(shape):
                    shape = source_shape
                elif not util._is_scalar_shape(source_shape):
                    raise ShapeInferenceError(f"Step {element.get_path_str()} received incompatible input shapes {[source.shape for source in known_sources]}")
            if source.dtype is not None:
                if dtype is None:
                    dtype = source.dtype
                elif dtype is not None and dtype != source.dtype:
                    raise_dtype_warning = True
                    dtype = util_jax.cfg["jdtype"]
        if raise_dtype_warning:
            warnings.warn(f"Step {element.get_path_str()} received incompatible input types {[source.dtype for source in known_sources]}.", TypeInferenceWarning, stacklevel=6)
        if dtype is None:
            dtype = dtype = util_jax.cfg["jdtype"]
        return shape, dtype

    def _collect_compile_info(self, circuit : Circuit) -> CompileInfo:
        """Gather compiled elements, endpoints, and kernels."""
        compile_info = CompileInfo(
            circuit=circuit,
            compiled_elements={},
            dynamic=[],
            static=[],
            sources=[],
            sinks=[],
            kernel_map={},
        )

        for element_name, element in circuit.element_map.items():
            if not element.is_compiled:
                continue

            element_ref = ElementRef(element)
            path = element_ref.path
            if isinstance(element, Circuit):
                child_info = self.local_compile_info[element]
                for child_ref in child_info.compiled_elements.values():
                    compile_info.compiled_elements[child_ref.path] = child_ref
                for child_path, child_kernel in child_info.kernel_map.items():
                    compile_info.kernel_map[child_path] = child_kernel

            compile_info.compiled_elements[path] = element_ref
            compile_info.kernel_map[path] = element.compute_kernel
            if element.is_dynamic:
                compile_info.dynamic.append(element_ref)
            else:
                compile_info.static.append(element_ref)
            if element.is_source:
                compile_info.sources.append(element_ref)
            if element.is_sink:
                compile_info.sinks.append(element_ref)

            if isinstance(element, Circuit):
                child_info = self.local_compile_info[element]
                for child_ref in child_info.dynamic:
                    compile_info.dynamic.append(child_ref)
                for child_ref in child_info.static:
                    compile_info.static.append(child_ref)
                for child_ref in child_info.sources:
                    compile_info.sources.append(child_ref)
                for child_ref in child_info.sinks:
                    compile_info.sinks.append(child_ref)

        return compile_info

    def _refresh_compile_info(self, circuit : Circuit):
        """Rebuild and cache CompileInfo for a circuit after inference changes."""
        self.local_compile_info[circuit] = self._collect_compile_info(circuit)
        return self.local_compile_info[circuit]

    def _gather_uncompiled_elements(self, circuit : Circuit) -> list[Element]:
        uncompiled_elements = []
        for element in circuit.element_map.values():
            if not element.is_compiled:
                uncompiled_elements.append(element)
            if isinstance(element, Circuit):
                uncompiled_elements += self._gather_uncompiled_elements(element)
        return uncompiled_elements
    
def _trace_compile_failure_sources(uncompiled_elements):
    """Traces the sources of why uncompiled elements didn't compile."""
    import networkx as nx
    # build graph for each element
    # graph with only one node and no edges are sources
    # group those elements with same source -> they likely failed to compile because of this
    # log the traces from the elements to the source

    # a element is a source if incoming slots are compiled (if existing) but the step is not compiled anyway
    # this should actually not really happen, since the api should catch this before it tries to compile, i guess..
    failure_trace_graph = nx.DiGraph()
    known_failure_sources = {}
    all_sources = []

    for element in uncompiled_elements:
        current_traced_element = element
        source_found = False
        known_failure_sources[element] = []
        while not source_found:
            failure_reasons = _check_direct_failure_source(current_traced_element)
            if failure_reasons is None:
                failure_trace_graph.add_node(current_traced_element)
                known_failure_sources[element].append(current_traced_element)
                all_sources.append(current_traced_element)
                source_found = True
            
            elif len(failure_reasons) == 1:
                failure_trace_graph.add_edge(current_traced_element, failure_reasons[0])
                current_traced_element = failure_reasons[0]

            elif len(failure_reasons) >= 1:
                # do some magic stuff here
                continue
        
    element_traces = []
    for element in uncompiled_elements:
        for failure_source in known_failure_sources[element]:
            element_traces.append(nx.shortest_path(failure_trace_graph, source=element, target=failure_source))
    
    element_traces.sort(key=lambda x: len(x))
    return element_traces, all_sources

def _check_direct_failure_source(element : Element) -> list[Element] | None:
    if element.is_compiled:
        raise CompilerError(f"The element {element.get_path_str()} is marked as part of a failure trace but is compiled, which is not possible.")
    
    incoming_slots = [element.parent_circuit.connection_map_reversed[element.get_local_circuit_id() + '.' + in_slot_id] for in_slot_id in element.input_slot_map.keys()] # check if the incoming steps are compiled and mark as failure surce if not (or at least if their slots are uncompiled)

    incoming_elements = [incoming_slot[0].parent for incoming_slot in incoming_slots]
    if hasattr(element, 'buffer_map'):
        buffer = element.buffer_map.values()
    output_slots = element.output_slot_map.values()

    failure_sources = None
    for incoming_element in incoming_elements:
        if not incoming_element.is_compiled:
            failure_sources = [incoming_element] if failure_sources is None else failure_sources.append(incoming_element)
            
    return failure_sources


def _changed_and_not_none(old: Any, new: Any) -> bool:
    return (old != new) and (new is not None)

def _slot_changed(slot, new_shape, new_dtype):
    if _changed_and_not_none(old=slot.shape, new=new_shape) or _changed_and_not_none(old=slot.dtype, new=new_dtype):
        slot.shape = new_shape
        slot.dtype = new_dtype
        return True
    return False
