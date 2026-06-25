from __future__ import annotations
import logging
from dataclasses import dataclass
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
            failed_elements = _gather_uncompiled_elements(circuit=circuit)
            report = _format_compile_failure_report(circuit, failed_elements)
            logger.error(report)
            raise CompilerError(report) from e

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
                changed = (
                    _changed_and_not_none(buffer.shape, shape)
                    or _changed_and_not_none(buffer.dtype, dtype)
                    or _changed_and_not_none(buffer.permanent, permanent)
                )
                buffer.shape = shape
                buffer.dtype = dtype
                buffer.permanent = permanent
                self._check_buffer_compiled(buffer)
                buffer_updated |= changed
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
  

def _changed_and_not_none(old: Any, new: Any) -> bool:
    return (old != new) and (new is not None)

def _slot_changed(slot, new_shape, new_dtype):
    if _changed_and_not_none(old=slot.shape, new=new_shape) or _changed_and_not_none(old=slot.dtype, new=new_dtype):
        slot.shape = new_shape
        slot.dtype = new_dtype
        return True
    return False










######################## Analyze compilation failures and generate failure code. #########################

@dataclass
class _CompileFailure:
    """Local compilation state for one uncompiled element."""

    upstream: tuple[Element, ...]
    codes: set[str]
    is_local_source: bool

def _gather_uncompiled_elements(circuit : Circuit) -> list[Element]:
    uncompiled_elements = []
    for element in circuit.element_map.values():
        if not element.is_compiled:
            uncompiled_elements.append(element)
        if isinstance(element, Circuit):
            uncompiled_elements += _gather_uncompiled_elements(element)
    return uncompiled_elements
  
def _trace_compile_failure_sources(
        uncompiled_elements: list[Element],
    ) -> tuple[list[list[Element]], list[Element], dict[Element, _CompileFailure], list[list[Element]]]:
    """Find all failed dependency paths, including branches and feedback loops.

    Edges point from an uncompiled element to the uncompiled upstream elements
    whose unresolved output prevents it from compiling. Returned traces point in
    user-facing order: likely source -> downstream affected element.
    """
    failed_elements = _unique_elements(uncompiled_elements)
    failed_set = set(failed_elements)
    failures = {
        element: _check_direct_failure_source(element, failed_set)
        for element in failed_elements
    }

    sources = {
        element
        for element, failure in failures.items()
        if failure.is_local_source or not failure.upstream
    }
    cycle_components = _find_failure_cycles(failures)

    # A feedback component is only a likely source if it cannot already reach
    # another concrete source through an upstream dependency.
    cycles = []
    for component in cycle_components:
        if any(_can_reach_failure_source(element, failures, sources, set()) for element in component):
            continue
        for element in component:
            failures[element].codes.add("CYCLIC_FAILURE_DEPENDENCY")
            sources.add(element)
        cycles.append(component)

    traces = []
    for element in failed_elements:
        for trace_to_source in _traces_to_failure_sources(element, failures, sources, ()):
            traces.append(list(reversed(trace_to_source)))

    unique_traces = []
    seen_traces = set()
    for trace in traces:
        trace_key = tuple(trace)
        if trace_key not in seen_traces:
            seen_traces.add(trace_key)
            unique_traces.append(trace)

    unique_traces.sort(key=lambda trace: (len(trace), _element_path(trace[0]), _element_path(trace[-1])))
    ordered_sources = sorted(sources, key=_element_path)
    return unique_traces, ordered_sources, failures, cycles


def _check_direct_failure_source(
        element: Element,
        failed_elements: set[Element] | None = None,
    ) -> _CompileFailure:
    """Classify an uncompiled element and its immediate failed prerequisites."""
    if element.is_compiled:
        raise CompilerError(
            f"The element {element.get_path_str()} is marked as part of a failure trace "
            "but is compiled."
        )

    failed_elements = failed_elements or set()
    codes = set()
    upstream = set()
    has_blocking_input = element.needs_input_connections and not element.is_source

    if has_blocking_input:
        for slot_id, input_slot in element.input_slot_map.items():
            incoming_slots = element.parent_circuit.connection_map_reversed.get(
                input_slot.get_local_circuit_id(),
                [],
            )
            if input_slot.is_compiled:
                continue

            if not incoming_slots:
                codes.add(f"INPUT_SLOT_UNCONNECTED:{slot_id}")
            else:
                codes.update(_slot_failure_codes("INPUT_SLOT", slot_id, input_slot))
                for source_slot in incoming_slots:
                    source_element = source_slot.parent
                    if source_element in failed_elements and not source_element.is_compiled:
                        upstream.add(source_element)

    if isinstance(element, Circuit):
        failed_children = [
            child
            for child in element.element_map.values()
            if child in failed_elements and not child.is_compiled
        ]
        if failed_children:
            codes.add("CHILD_ELEMENT_UNCOMPILED")
            upstream.update(failed_children)

    for slot_id, output_slot in element.output_slot_map.items():
        if not output_slot.is_compiled:
            codes.update(_slot_failure_codes("OUTPUT_SLOT", slot_id, output_slot))

    for buffer_id, buffer in getattr(element, "buffer_map", {}).items():
        if not buffer.is_compiled:
            codes.update(_buffer_failure_codes(buffer_id, buffer))

    # An element whose required inputs are known but whose own outputs/buffers
    # are unknown cannot be explained by an upstream failure. It is a likely
    # failure source itself.
    own_spec_failure = any(
        code.startswith(("OUTPUT_SLOT_", "BUFFER_"))
        for code in codes
    )
    all_required_inputs_compiled = (
        not has_blocking_input
        or all(slot.is_compiled for slot in element.input_slot_map.values())
    )
    is_local_source = (
        not upstream
        and (
            own_spec_failure
            or any(code.startswith("INPUT_SLOT_UNCONNECTED:") for code in codes)
            or (all_required_inputs_compiled and bool(codes))
        )
    )

    if upstream:
        codes.add("UPSTREAM_ELEMENT_UNCOMPILED")
    if not codes:
        codes.add("COMPILE_STATE_UNRESOLVED")
        is_local_source = not upstream

    return _CompileFailure(
        upstream=tuple(sorted(upstream, key=_element_path)),
        codes=codes,
        is_local_source=is_local_source,
    )


def _format_compile_failure_report(circuit: Circuit, failed_elements: list[Element]) -> str:
    """Build the user-facing compilation failure report."""
    traces, sources, failures, cycles = _trace_compile_failure_sources(failed_elements)
    lines = [f"The circuit '{circuit.get_local_circuit_id()}' could not be compiled."]

    lines.append("\nLikely failure sources:")
    for source in sources:
        codes = ", ".join(sorted(failures[source].codes))
        lines.append(f"- {_element_label(source)} [{codes}]")

    lines.append("\nUncompiled elements:")
    for element in sorted(_unique_elements(failed_elements), key=_element_path):
        codes = ", ".join(sorted(failures[element].codes))
        lines.append(f"- {_element_label(element)} [{codes}]")

    if traces:
        lines.append("\nFailure propagation:")
        for trace in traces:
            lines.append(f"- {' -> '.join(_element_label(element) for element in trace)}")

    if cycles:
        lines.append("\nFailure dependency cycles:")
        for cycle in cycles:
            lines.append(
                f"- {' -> '.join(_element_label(element) for element in _cycle_trace(cycle, failures))}"
            )

    return "\n".join(lines)


def _slot_failure_codes(prefix: str, slot_id: str, slot: Slot) -> set[str]:
    codes = set()
    if slot.shape is None:
        codes.add(f"{prefix}_SHAPE_UNRESOLVED:{slot_id}")
    if slot.dtype is None:
        codes.add(f"{prefix}_DTYPE_UNRESOLVED:{slot_id}")
    return codes


def _buffer_failure_codes(buffer_id: str, buffer: Buffer) -> set[str]:
    codes = set()
    if buffer.shape is None:
        codes.add(f"BUFFER_SHAPE_UNRESOLVED:{buffer_id}")
    if buffer.dtype is None:
        codes.add(f"BUFFER_DTYPE_UNRESOLVED:{buffer_id}")
    if buffer.permanent is None:
        codes.add(f"BUFFER_PERMANENCE_UNRESOLVED:{buffer_id}")
    return codes


def _find_failure_cycles(failures: dict[Element, _CompileFailure]) -> list[list[Element]]:
    """Return strongly connected components that represent dependency loops."""
    index = 0
    indices = {}
    lowlinks = {}
    stack = []
    on_stack = set()
    components = []

    def visit(element: Element):
        nonlocal index
        indices[element] = index
        lowlinks[element] = index
        index += 1
        stack.append(element)
        on_stack.add(element)

        for upstream in failures[element].upstream:
            if upstream not in indices:
                visit(upstream)
                lowlinks[element] = min(lowlinks[element], lowlinks[upstream])
            elif upstream in on_stack:
                lowlinks[element] = min(lowlinks[element], indices[upstream])

        if lowlinks[element] != indices[element]:
            return

        component = []
        while True:
            upstream = stack.pop()
            on_stack.remove(upstream)
            component.append(upstream)
            if upstream is element:
                break

        if len(component) > 1 or element in failures[element].upstream:
            components.append(sorted(component, key=_element_path))

    for element in failures:
        if element not in indices:
            visit(element)
    return components


def _can_reach_failure_source(
        element: Element,
        failures: dict[Element, _CompileFailure],
        sources: set[Element],
        visited: set[Element],
    ) -> bool:
    if element in visited:
        return False
    if element in sources:
        return True
    visited = visited | {element}
    return any(
        _can_reach_failure_source(upstream, failures, sources, visited)
        for upstream in failures[element].upstream
    )


def _traces_to_failure_sources(
        element: Element,
        failures: dict[Element, _CompileFailure],
        sources: set[Element],
        path: tuple[Element, ...],
    ) -> list[tuple[Element, ...]]:
    """Return every acyclic path from an affected element to a likely source."""
    if element in path:
        return []
    current_path = path + (element,)
    if element in sources:
        return [current_path]

    traces = []
    for upstream in failures[element].upstream:
        traces.extend(_traces_to_failure_sources(upstream, failures, sources, current_path))
    return traces


def _cycle_trace(
        component: list[Element],
        failures: dict[Element, _CompileFailure],
    ) -> list[Element]:
    """Return one real directed loop from a strongly connected component."""
    component_set = set(component)
    path = []
    positions = {}
    current = component[0]

    while current not in positions:
        positions[current] = len(path)
        path.append(current)
        current = next(
            upstream
            for upstream in failures[current].upstream
            if upstream in component_set
        )

    return path[positions[current]:] + [current]


def _unique_elements(elements: list[Element]) -> list[Element]:
    return list(dict.fromkeys(elements))


def _element_path(element: Element) -> str:
    return element.get_path_str() or element.get_local_circuit_id()


def _element_label(element: Element) -> str:
    return f"{type(element).__name__}({_element_path(element)})"
