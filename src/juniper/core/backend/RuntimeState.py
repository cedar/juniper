from __future__ import annotations
import logging
from typing import Any
import warnings
import os

from .DataClasses import ElementRef
from .DataClasses import CompileInfo
from .DataClasses import ElementPath
from .DataClasses import StateTree

from .Exceptions import LoadBufferError
from .Exceptions import SaveBufferError
from .Warnings import LoadBufferWarning

import json
import jax.numpy as jnp
import numpy as np

from ..frontend.Circuit import Circuit
from ..frontend.Element import Element
from ..frontend.Connectable import Connectable
from ...dft.NeuralField import NeuralField
from ...util import util
from ...util import util_jax
from ...util.util_jax import constant
from ...util.util_jax import zeros


logger = logging.getLogger(__name__)
class RuntimeState:
    """Small wrapper around the JAX state tree.

    RuntimeState hides path-key handling from Engine. Engine can work with
    ElementRef objects while this class reads and writes the matching
    state entry.

    shape:
        ref.path = ("sub", "field")
        get(ref) -> state_tree[("sub", "field")]
        set(ref, x) -> state_tree[("sub", "field")] = x
    """

    def __init__(self, state_tree: StateTree):
        """Store the flat state tree used by the jitted tick function."""
        self.state_tree = state_tree

    @classmethod
    def from_compile_info(cls, compile_info: CompileInfo) -> RuntimeState:
        """Allocate the initial runtime state from compiled slot/buffer specs."""
        return cls(_init_state(compile_info))

    def get_specs(self) -> StateTree:
        """Return the shape and dtype for each element, slot and buffer in the state tree"""
        specs = {}
        for key, value in self.state_tree.items():
            specs[key] = {}
            for id, jnparray in value.items():
                specs[key][id] = (jnparray.shape, jnparray.dtype)
        return specs

    def copy(self) -> RuntimeState:
        """Copy the top-level tree."""
        return RuntimeState(self.state_tree.copy())

    def trace_state_tree(self, path: ElementPath) -> StateTree:
        """Return the flat state entry for a path."""
        return self.state_tree[path]

    def get(self, ref: ElementRef) -> StateTree:
        """Return the state entry owned by an element reference."""
        return self.trace_state_tree(ref.path)
    
    def get_parent(self, ref: ElementRef) -> StateTree:
        """Flat state has no parent subtree; return the top-level tree."""
        return self.state_tree

    def set(self, ref: ElementRef, step_state: StateTree) -> None:
        """Replace the state entry owned by an element reference."""
        self.state_tree[ref.path] = step_state

    def read_slot(self, ref: ElementRef, slot_id: str = util.DEFAULT_OUTPUT_SLOT) -> np.ndarray:
        """Copy a slot from device/runtime state back to a NumPy array."""
        return np.array(self.get(ref)[slot_id])

    def write_source_output(self, ref: ElementRef, data: Any) -> None:
        """Write CPU-side source data into the source's default output slot."""
        step_state = dict(self.get(ref))
        output_slot = ref.element.output_slot_map[util.DEFAULT_OUTPUT_SLOT]
        step_state[util.DEFAULT_OUTPUT_SLOT] = jnp.array(data, dtype=output_slot.dtype)
        self.set(ref, step_state)

    def record(self, compile_info: CompileInfo, target: Connectable | str) -> np.ndarray:
        """Read a recording target such as 'field' or 'field.out0'."""
        if isinstance(target, str):
            path = tuple(target.split("."))
            if path in compile_info.compiled_elements:
                ref = compile_info.ref_at(path)
                slot_id = util.DEFAULT_OUTPUT_SLOT
            else:
                path_str, slot_id = target.rsplit(".", 1) if "." in target else (target, util.DEFAULT_OUTPUT_SLOT)
                ref = compile_info.ref_at(tuple(path_str.split(".")))
        elif isinstance(target, Connectable):
            slot = target.get_slot_from_connectable(target)
            ref = ElementRef(slot.parent)
            slot_id = slot.get_slot_id()

        return self.read_slot(ref, slot_id)


def _init_state(compile_info: CompileInfo) -> StateTree:
    """Allocate one state for each compiled element."""
    state = {}
    for ref in compile_info.compiled_elements.values():
        element = ref.element
        if isinstance(element, Circuit):
            state[ref.path] = _init_circuit_state(ref.element)
        else:
            state[ref.path] = _init_step_state(ref.element)
    return state


def _init_circuit_state(circuit: Circuit) -> StateTree:
    """Allocate output slots for a compiled circuit element."""
    state = {}
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
    data_file = util_jax.cfg["arch_file_path"] + compile_info.circuit.get_local_circuit_id() + ".data"
    if os.path.isfile(data_file):
        with open(data_file, "r") as f:
            tree = json.load(f)
    else:
        warnings.warn("Could not find data file to load permanent buffer.", LoadBufferWarning)
        return {}

    loaded_buffer = {}
    for path_str, step_tree in tree.items():
        path = tuple(path_str.split("."))
        try:
            if path not in compile_info.compiled_elements:
                raise LoadBufferError(f"No compiled step at path {path_str}")
            if "BUFFER" not in step_tree:
                raise LoadBufferError(f"Invalid buffer format. Expected BUFFER, got {step_tree.keys()}")

            ref = compile_info.ref_at(path)
            buffer_map = getattr(ref.element, "buffer_map", {})
            step_state = dict(runtime_state.get(ref))
            loaded_step_buffer = {}

            for buffer_id, buffer_data in step_tree["BUFFER"].items():
                if buffer_id not in buffer_map:
                    raise LoadBufferError(f"Step {path_str} has no buffer '{buffer_id}'")

                buffer = buffer_map[buffer_id]
                loaded_array = jnp.array(buffer_data, dtype=buffer.dtype or util_jax.cfg["jdtype"])
                step_state[buffer_id] = loaded_array
                loaded_step_buffer[buffer_id] = loaded_array

            runtime_state.set(ref, step_state)
            loaded_buffer[path_str] = loaded_step_buffer
        except Exception as e:
            logger.error("-- Error during Engine::load_buffers('{data_file}') -- Buffer for step " + path_str + " could not be loaded.")
            logger.error(e)
            warnings.warn("Buffer for step " + path_str + " could not be loaded")

    return loaded_buffer

def save_permanent_buffers(compile_info: CompileInfo, runtime_state: RuntimeState) -> None:
    """Save persistant buffer to disk."""
    tree = {}
    try:
        for ref in compile_info.compiled_elements.values():
            buffers = {}
            step_state = runtime_state.get(ref)
            for buffer_id, buffer in getattr(ref.element, "buffer_map", {}).items():
                if buffer.permanent:
                    buffers[buffer_id] = np.array(step_state[buffer_id]).tolist()
            if len(buffers) > 0:
                tree[".".join(ref.path)] = {"BUFFER": buffers}
    except Exception as e:
        raise SaveBufferError("Failed to construct buffer tree.") from e

    try:
        if len(tree) > 0:
            data_file = util_jax.cfg["arch_file_path"] + compile_info.circuit.get_local_circuit_id() + ".data"
            with open(data_file, "w") as f:
                f.write(json.dumps(tree, indent=4))
    except Exception as e:
        raise SaveBufferError("Failed to save buffer tree.") from e
