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


@dataclass(frozen=True)
class ElementRef:
    path: tuple[str, ...]
    element: Element

    @property
    def name(self) -> str:
        return self.element.get_name()


@dataclass(frozen=True)
class StateSpec:
    ref: ElementRef
    kind: str
    input_slots: dict[str, Any]
    output_slots: dict[str, Any]
    buffer_map: dict[str, Any]

    @property
    def element(self) -> Element:
        return self.ref.element

    @property
    def path(self) -> tuple[str, ...]:
        return self.ref.path


@dataclass
class CompileInfo:
    circuit: Circuit
    compiled_elements: dict[tuple[str, ...], ElementRef]
    dynamic: list[ElementRef]
    static: list[ElementRef]
    sources: list[ElementRef]
    sinks: list[ElementRef]
    sub_processes: list[ElementRef]
    state_specs: dict[tuple[str, ...], StateSpec]
    kernel_map: KernelMap
    children: dict[str, CompileInfo]

    def ref_at(self, path: tuple[str, ...]) -> ElementRef:
        return self.compiled_elements[path]

    def spec_at(self, path: tuple[str, ...]) -> StateSpec:
        return self.state_specs[path]

    def dynamic_leaf_paths(self) -> set[tuple[str, ...]]:
        return {
            ref.path
            for ref in self.dynamic
            if not isinstance(ref.element, Circuit)
        }

    def runtime_connections(self) -> list[ElementRef]:
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
    def __init__(self, tree: StateTree):
        self.tree = tree

    @classmethod
    def from_compile_info(cls, compile_info: CompileInfo) -> RuntimeState:
        return cls(_init_circuit_state(compile_info.circuit))

    def copy(self) -> RuntimeState:
        return RuntimeState(self.tree.copy())

    def get(self, ref: ElementRef) -> StateTree:
        state = self.tree
        for name in ref.path:
            state = state[name]
        return state

    def set(self, ref: ElementRef, step_state: StateTree) -> None:
        if len(ref.path) == 1:
            self.tree[ref.path[0]] = step_state
            return

        state = self.tree
        for name in ref.path[:-1]:
            state = state[name]
        state[ref.path[-1]] = step_state

    def read_slot(self, ref: ElementRef, slot_id: str = util.DEFAULT_OUTPUT_SLOT) -> np.ndarray:
        return np.array(self.get(ref)[slot_id])

    def write_source_output(self, ref: ElementRef, data: Any) -> None:
        step_state = dict(self.get(ref))
        output_slot = ref.element.output_slot_map[util.DEFAULT_OUTPUT_SLOT]
        step_state[util.DEFAULT_OUTPUT_SLOT] = jnp.array(data, dtype=output_slot.dtype)
        self.set(ref, step_state)

    def record(self, compile_info: CompileInfo, target: str) -> np.ndarray:
        path_str, slot_id = target.rsplit(".", 1) if "." in target else (target, util.DEFAULT_OUTPUT_SLOT)
        ref = compile_info.ref_at(tuple(path_str.split(".")))
        return self.read_slot(ref, slot_id)


def _init_circuit_state(circuit: Circuit) -> StateTree:
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
    data_file = util_jax.cfg["arch_file_path"] + compile_info.circuit.get_name() + ".data"
    with open(data_file, "r") as f:
        tree = json.load(f)

    return _load_permanent_buffers_from_tree(compile_info, runtime_state, tree, data_file)


def _load_permanent_buffers_from_tree(
    compile_info: CompileInfo,
    runtime_state: RuntimeState,
    tree: dict[str, Any],
    data_file: str,
) -> dict[str, dict[str, Any]]:
    loaded_buffer = {}
    for path_str, step_tree in tree.items():
        path = tuple(path_str.split("."))
        try:
            if path not in compile_info.state_specs:
                raise Exception(f"No compiled step at path {path_str}")
            if "BUFFER" not in step_tree:
                raise Exception(f"Invalid buffer format. Expected BUFFER, got {step_tree.keys()}")

            spec = compile_info.spec_at(path)
            step_state = dict(runtime_state.get(spec.ref))
            loaded_step_buffer = {}

            for buffer_id, buffer_data in step_tree["BUFFER"].items():
                if buffer_id not in spec.buffer_map:
                    raise Exception(f"Step {path_str} has no buffer '{buffer_id}'")

                buffer = spec.buffer_map[buffer_id]
                loaded_array = jnp.array(buffer_data, dtype=buffer.dtype or util_jax.cfg["jdtype"])
                step_state[buffer_id] = loaded_array
                loaded_step_buffer[buffer_id] = loaded_array

            runtime_state.set(spec.ref, step_state)
            loaded_buffer[path_str] = loaded_step_buffer
        except Exception as e:
            print(f"-- Error during Engine::load_buffers('{data_file}') --")
            print(e)
            print("Buffer for step " + path_str + " could not be loaded")

    return loaded_buffer


def save_permanent_buffers(compile_info: CompileInfo, runtime_state: RuntimeState) -> None:
    tree = {}
    for spec in compile_info.state_specs.values():
        buffers = {}
        step_state = runtime_state.get(spec.ref)
        for buffer_id, buffer in spec.buffer_map.items():
            if getattr(buffer, "permanent", False):
                buffers[buffer_id] = np.array(step_state[buffer_id]).tolist()
        if len(buffers) > 0:
            tree[".".join(spec.path)] = {"BUFFER": buffers}

    if len(tree) > 0:
        data_file = util_jax.cfg["arch_file_path"] + compile_info.circuit.get_name() + ".data"
        with open(data_file, "w") as f:
            f.write(json.dumps(tree, indent=4))
