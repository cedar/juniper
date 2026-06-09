from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json
import jax.numpy as jnp
import numpy as np

from .configurables.Circuit import Circuit
from .configurables.Element import Element
from .dft.NeuralField import NeuralField
from .util import util
from .util import util_jax
from .util.util_jax import constant
from .util.util_jax import zeros

StateTree = dict[str, Any]
KernelMap = dict[str, dict[str, Any]]
Path = tuple[str, ...]


@dataclass(frozen=True)
class ElementRef:
    """Stable handle to an element inside the nested runtime state tree."""

    path: Path
    element: Element

    @property
    def name(self) -> str:
        """Return the element name."""
        return self.element.get_name()


@dataclass
class CompileInfo:
    """Runtime result of compilation.

    This object is the contract between Compiler and Engine: graph traversal,
    state locations, kernels, IO endpoints, and process ownership are gathered
    here so Engine does not need to inspect the user-defined circuit object directly.

    shape:
        compiled_elements[("sub", "field")] -> ElementRef(path, field)
        sources/sinks/dynamic/static -> lists of ElementRef
        kernel_map -> nested tree that mirrors Circuit.compute_kernel
    """

    circuit: Circuit
    compiled_elements: dict[Path, ElementRef]
    dynamic: list[ElementRef]
    static: list[ElementRef]
    sources: list[ElementRef]
    sinks: list[ElementRef]
    sub_processes: list[ElementRef]
    kernel_map: KernelMap

    def ref_at(self, path: Path) -> ElementRef:
        """Look up the compiled element handle for a path."""
        return self.compiled_elements[path]

    def dynamic_step_paths(self) -> set[Path]:
        """Return paths to dynamic steps that need fresh PRNG keys."""
        return {
            ref.path
            for ref in self.dynamic
            if not isinstance(ref.element, Circuit)
        }

    def runtime_connections(self) -> list[ElementRef]:
        """Return handles for each source, sink, or managed process once. used for open/close connections."""
        refs = []
        seen = set()
        for group in (self.sources, self.sinks, self.sub_processes):
            for ref in group:
                element_id = id(ref.element)
                if element_id in seen:
                    continue
                seen.add(element_id)
                refs.append(ref)
        return refs


class RuntimeState:
    """Small wrapper around the nested JAX state tree.

    RuntimeState hides path traversal from Engine. Engine can work with
    ElementRef objects while this class reads and writes the matching subtree.

    shape:
        ref.path = ("sub", "field")
        get(ref) -> state_tree["sub"]["field"]
        set(ref, x) -> state_tree["sub"]["field"] = x
    """

    def __init__(self, state_tree: StateTree):
        """Store the nested state tree used by the jitted tick function."""
        self.state_tree = state_tree

    @classmethod
    def from_compile_info(cls, compile_info: CompileInfo) -> RuntimeState:
        """Allocate the initial runtime state from compiled slot/buffer specs."""
        return cls(_init_circuit_state(compile_info.circuit))

    def copy(self) -> RuntimeState:
        """Copy the top-level tree."""
        return RuntimeState(self.state_tree.copy())

    def trace_state_tree(self, path: Path) -> StateTree:
        """Traces the specified path in the state tree and returns the sub_state."""
        state = self.state_tree
        for name in path:
            state = state[name]
        return state

    def get(self, ref: ElementRef) -> StateTree:
        """Return the state subtree owned by an element reference."""
        return self.trace_state_tree(ref.path)
    
    def get_parent(self, ref: ElementRef) -> StateTree:
        """Returns parents state tree of the element reference."""
        return self.trace_state_tree(ref.path[:-1])

    def set(self, ref: ElementRef, step_state: StateTree) -> None:
        """Replace the state subtree owned by an element reference."""
        if len(ref.path) == 1:
            self.state_tree[ref.path[0]] = step_state
            return

        state = self.get_parent(ref)
        state[ref.path[-1]] = step_state

    def read_slot(self, ref: ElementRef, slot_id: str = util.DEFAULT_OUTPUT_SLOT) -> np.ndarray:
        """Copy a slot from device/runtime state back to a NumPy array."""
        return np.array(self.get(ref)[slot_id])

    def write_source_output(self, ref: ElementRef, data: Any) -> None:
        """Write CPU-side source data into the source's default output slot."""
        step_state = dict(self.get(ref))
        output_slot = ref.element.output_slot_map[util.DEFAULT_OUTPUT_SLOT]
        step_state[util.DEFAULT_OUTPUT_SLOT] = jnp.array(data, dtype=output_slot.dtype)
        self.set(ref, step_state)

    def record(self, compile_info: CompileInfo, target: str) -> np.ndarray:
        """Read a recording target such as 'field' or 'field.out0'."""
        path_str, slot_id = target.rsplit(".", 1) if "." in target else (target, util.DEFAULT_OUTPUT_SLOT)
        ref = compile_info.ref_at(tuple(path_str.split(".")))
        return self.read_slot(ref, slot_id)


def _init_circuit_state(circuit: Circuit) -> StateTree:
    """Recursively allocate output slots and buffers for a compiled circuit."""
    state = {}
    for element_name, element in circuit.element_map.items():
        if element is not circuit:
            if isinstance(element, Circuit):
                state[element_name] = _init_circuit_state(element)
            else:
                state[element_name] = _init_step_state(element)
    for slot_id, slot in circuit.output_slot_map.items():
        state[slot_id] = zeros(slot.shape, slot.dtype)
    return state


def _init_step_state(element: Element) -> StateTree:
    """Allocate the slots and buffer for a compiled step."""
    state = {}
    for slot_id, slot in element.output_slot_map.items():
        state[slot_id] = zeros(slot.shape, slot.dtype)
    for buffer_id, buffer in getattr(element, "buffer_map", {}).items():
        if isinstance(element, NeuralField):
            state[buffer_id] = constant(buffer.shape, buffer.dtype, element._params["resting_level"])
        else:
            state[buffer_id] = zeros(buffer.shape, buffer.dtype)
    return state


def load_permanent_buffers(compile_info: CompileInfo, runtime_state: RuntimeState) -> dict[str, dict[str, Any]]:
    """Load permanent buffers from the circuit's data file into runtime state."""
    data_file = util_jax.cfg["arch_file_path"] + compile_info.circuit.get_name() + ".data"
    with open(data_file, "r") as f:
        tree = json.load(f)

    loaded_buffer = {}
    for path_str, step_tree in tree.items():
        path = tuple(path_str.split("."))
        try:
            if path not in compile_info.compiled_elements:
                raise Exception(f"No compiled step at path {path_str}")
            if "BUFFER" not in step_tree:
                raise Exception(f"Invalid buffer format. Expected BUFFER, got {step_tree.keys()}")

            ref = compile_info.ref_at(path)
            buffer_map = getattr(ref.element, "buffer_map", {})
            step_state = dict(runtime_state.get(ref))
            loaded_step_buffer = {}

            for buffer_id, buffer_data in step_tree["BUFFER"].items():
                if buffer_id not in buffer_map:
                    raise Exception(f"Step {path_str} has no buffer '{buffer_id}'")

                buffer = buffer_map[buffer_id]
                loaded_array = jnp.array(buffer_data, dtype=buffer.dtype or util_jax.cfg["jdtype"])
                step_state[buffer_id] = loaded_array
                loaded_step_buffer[buffer_id] = loaded_array

            runtime_state.set(ref, step_state)
            loaded_buffer[path_str] = loaded_step_buffer
        except Exception as e:
            print(f"-- Error during Engine::load_buffers('{data_file}') --")
            print(e)
            print("Buffer for step " + path_str + " could not be loaded")

    return loaded_buffer

def save_permanent_buffers(compile_info: CompileInfo, runtime_state: RuntimeState) -> None:
    """Save persistant buffer to disk."""
    tree = {}
    for ref in compile_info.compiled_elements.values():
        buffers = {}
        step_state = runtime_state.get(ref)
        for buffer_id, buffer in getattr(ref.element, "buffer_map", {}).items():
            if buffer.permanent:
                buffers[buffer_id] = np.array(step_state[buffer_id]).tolist()
        if len(buffers) > 0:
            tree[".".join(ref.path)] = {"BUFFER": buffers}

    if len(tree) > 0:
        data_file = util_jax.cfg["arch_file_path"] + compile_info.circuit.get_name() + ".data"
        with open(data_file, "w") as f:
            f.write(json.dumps(tree, indent=4))
