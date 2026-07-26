from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ...util import util_jax
from ...util.util import timer
from ..frontend.Circuit import Circuit
from .Compiler import compile
from .DataClasses import CompileInfo, RecKey, Recording, StateTree, TimingInfo
from .Exceptions import EngineError, NotCompiledError
from .RuntimeState import RuntimeState, load_permanent_buffers, save_permanent_buffers

JAXTRACECOUNTER = 1

logger = logging.getLogger(__name__)

@dataclass(eq=False)
class SimulationRuntime:
    """Mutable runtime bundle used by simulation helper functions."""

    compile_info: CompileInfo
    init_state: RuntimeState
    state: RuntimeState
    prng_tree: StateTree
    prng_slots: list
    static_prng_key: Any

    @property
    def circuit(self) -> Circuit:
        return self.compile_info.circuit

    @property
    def kernel_map(self) -> dict:
        return self.compile_info.kernel_map

    @classmethod
    def from_circuit(cls, circuit: Circuit, static_prng_key: Any | None = None) -> SimulationRuntime:
        runtime_state, compile_info = compile(circuit)
        return cls.from_compiled_circuit(runtime_state, compile_info, static_prng_key=static_prng_key)

    @classmethod
    def from_compiled_circuit(
        cls,
        runtime_state: RuntimeState,
        compile_info: CompileInfo,
        static_prng_key: Any | None = None,
    ) -> SimulationRuntime:
        static_prng_key = util_jax.next_random_key() if static_prng_key is None else static_prng_key
        runtime = cls(
            compile_info=compile_info,
            init_state=runtime_state.copy(),
            state=runtime_state,
            prng_tree={},
            prng_slots=[],
            static_prng_key=static_prng_key,
        )
        init_prng(runtime)
        return runtime


def trace(runtime: SimulationRuntime, warmup : int = 0) -> StateTree:
    """Trace the jitted tick function once without mutating runtime state."""
    global JAXTRACECOUNTER
    _ensure_compiled(runtime)
    state_tree = _tick(runtime, runtime.state.state_tree, runtime.prng_tree)
    JAXTRACECOUNTER += 1
    if warmup > 0:
        _ = run_simulation(
            runtime,
            num_steps=warmup,
            steps_to_record=[],
            print_timing=False,
        )
    return state_tree


def init_prng(runtime: SimulationRuntime) -> tuple[StateTree, list]:
    """Initialize the runtime PRNG tree from its compile info."""
    runtime.prng_tree, runtime.prng_slots = util_jax.build_prng_tree(
        runtime.kernel_map,
        runtime.compile_info.dynamic_step_paths(),
        runtime.static_prng_key,
    )
    return runtime.prng_tree, runtime.prng_slots


def refresh_prng(runtime: SimulationRuntime) -> StateTree:
    """Refresh PRNG keys for dynamic steps and return the updated key tree."""
    runtime.prng_tree = util_jax.update_prng_tree(runtime.prng_tree, runtime.prng_slots)
    return runtime.prng_tree


def load_buffers(runtime: SimulationRuntime) -> dict[str, dict[str, Any]]:
    """Load permanent buffers into the runtime state."""
    _ensure_compiled(runtime)
    return load_permanent_buffers(runtime.compile_info, runtime.state)


def save_buffers(runtime: SimulationRuntime) -> None:
    """Save permanent buffers from the runtime state."""
    _ensure_compiled(runtime)
    save_permanent_buffers(runtime.compile_info, runtime.state)


def reset_state(runtime: SimulationRuntime) -> None:
    """Reset runtime state to the post-compilation initial state."""
    runtime.state = runtime.init_state.copy()


def close_connections(runtime: SimulationRuntime) -> None:
    """Close all runtime IO endpoints."""
    for ref in runtime.compile_info.gather_connections():
        ref.element.close()


def open_connections(runtime: SimulationRuntime) -> None:
    """Open all runtime IO endpoints."""
    for ref in runtime.compile_info.gather_connections():
        ref.element.open()


def run_simulation(
    runtime: SimulationRuntime,
    num_steps: int,
    steps_to_record: list[RecKey] | None = None,
    print_timing: bool = True,
    save_buffer: bool = False,
) -> tuple[Recording, TimingInfo]:
    """Run the simulation loop for a fixed number of ticks."""
    if steps_to_record is None:
        steps_to_record = []
    global JAXTRACECOUNTER
    _ensure_compiled(runtime)

    history = []
    key_timings = []
    tick_timings = []
    gpu_push_timings = []
    gpu_pull_timings = []

    for _ in range(num_steps):
        t_gpu_push, _ = timer(_push_sources)(runtime)
        gpu_push_timings.append(t_gpu_push)

        t_key, prng_keys = timer(refresh_prng)(runtime)
        key_timings.append(t_key)

        t_tick, state_tree = timer(_tick)(runtime, runtime.state.state_tree, prng_keys)
        JAXTRACECOUNTER += 1
        runtime.state.state_tree = state_tree
        tick_timings.append(t_tick)

        t_sink_pull, _ = timer(_pull_sinks)(runtime)
        t_recording_pull, data = timer(_pull_recordings)(runtime, steps_to_record)
        history.append(data)
        gpu_pull_timings.append(t_sink_pull + t_recording_pull)

    t_buffer_write = 0
    if save_buffer:
        t_buffer_write, _ = timer(save_buffers)(runtime)

    t_total = (
        np.sum(gpu_push_timings)
        + np.sum(key_timings)
        + np.sum(tick_timings)
        + np.sum(gpu_pull_timings)
        + t_buffer_write
    )
    timing_info = {
        "total": t_total,
        "prng": key_timings,
        "gpu_push": gpu_push_timings,
        "gpu_pull": gpu_pull_timings,
        "tick": tick_timings,
        "buffer": t_buffer_write,
        "num_steps": num_steps,
    }

    if print_timing:
        _print_timing(timing_info)

    return Recording(history, steps_to_record), timing_info

@partial(jax.jit, static_argnames=["runtime"])
def _tick(runtime: SimulationRuntime, state: StateTree, prng_keys: StateTree) -> StateTree:
    """Execute one compiled tick inside JAX."""
    new_state = state.copy()

    if JAXTRACECOUNTER > 1:
        logger.warning(
            "Jax retracing detected on Jax tick attempt number: "
            f"{JAXTRACECOUNTER}\nThis usually results from state drift due to poor shape "
            "and/or dtype definitions of external inputs or step definitions."
        )
    for element_path, kernel in runtime.kernel_map.items():
        ref = runtime.compile_info.compiled_elements[element_path]
        element = ref.element

        element_input = _gather_element_input(runtime, new_state, element)
        step_state = kernel(
            element_input,
            state[element_path],
            prng_key=prng_keys[element_path], prng_keys=prng_keys[element_path],
        )
        new_state[element_path] = _normalize_step_state(
            state[element_path],
            step_state,
            element,
        )

    return new_state

def _gather_element_input(runtime: SimulationRuntime, state: StateTree, element: Circuit) -> dict[str, Any]:
    """Build a state input dict for one element."""
    element_input = {}
    for slot_id, input_slot in element.input_slot_map.items():
        sources = element.parent.connection_map_reversed[input_slot.get_local_circuit_id()]
        element_input[slot_id] = _aggregate_slot_values(runtime, state, sources, element.input_aggregation)

    if isinstance(element, Circuit):
        for out_slot in element.output_slot_map.values():
            source_slots = element.connection_map_reversed[out_slot.get_local_circuit_id()]
            element_input[out_slot.get_slot_id()] = _aggregate_slot_values(
                runtime,
                state,
                source_slots,
                element.input_aggregation,
            )

    return element_input

def _aggregate_slot_values(
    runtime: SimulationRuntime,
    state: StateTree,
    slots: list[Any],
    aggregation: str,
) -> Any:
    """Resolve source slots in flat state and combine them for one input."""
    if len(slots) == 0:
        return 0

    value = _read_source_slot(runtime, state, slots[0])
    for slot in slots[1:]:
        slot_value = _read_source_slot(runtime, state, slot)
        if aggregation == "product":
            value = value * slot_value
        else:
            value = value + slot_value
    return value

def _read_source_slot(runtime: SimulationRuntime, state: StateTree, slot: Any) -> Any:
    """Read a source slot from flat state, following circuit input bridges."""
    source = slot.parent
    slot_id = slot.get_slot_id()

    if isinstance(source, Circuit) and slot_id in source.input_slot_map:
        sources = source.parent.connection_map_reversed[slot.get_local_circuit_id()]
        return _aggregate_slot_values(runtime, state, sources, source.input_aggregation)

    return state[source.get_path()][slot_id]

def _push_sources(runtime: SimulationRuntime) -> None:
    """Copy CPU-side source data into runtime state before a tick."""
    for ref in runtime.compile_info.sources:
        element = ref.element
        data = element.get_data()
        if data is None:
            continue
        runtime.state.write_source_output(ref, data)

def _pull_sinks(runtime: SimulationRuntime) -> None:
    """Copy sink outputs from runtime state back to their Python objects."""
    for ref in runtime.compile_info.sinks:
        ref.element.set_data(runtime.state.read_slot(ref))

def _pull_recordings(runtime: SimulationRuntime, steps_to_record: list[RecKey]) -> list[np.ndarray]:
    """Read requested recording targets from runtime state."""
    data = []
    for to_record in steps_to_record:
        data.append(runtime.state.record(runtime.compile_info, to_record))
    return data

def _print_timing(timing_info: TimingInfo) -> None:
    t_total = timing_info["total"]
    key_timings = timing_info["prng"]
    gpu_push_timings = timing_info["gpu_push"]
    gpu_pull_timings = timing_info["gpu_pull"]
    tick_timings = timing_info["tick"]
    t_buffer_write = timing_info["buffer"]
    num_steps = timing_info["num_steps"]

    ms_per_tick = 1000 * t_total / num_steps
    avg_gpu_push = np.mean(gpu_push_timings, axis=0)
    avg_tick = np.mean(tick_timings, axis=0)
    avg_gpu_pull = np.mean(gpu_pull_timings, axis=0)
    avg_prng = np.mean(key_timings, axis=0)

    print(f"{t_total:6.2f} s total duration [{num_steps} steps]")
    print(f"{ms_per_tick:6.2f} ms / time step")
    print(f"{(1000 * avg_gpu_push):6.2f} ms average time for gpu write operation")
    print(f"{(1000 * avg_prng):6.2f} ms average time for prng key generation")
    print(f"{(1000 * avg_tick):6.2f} ms average time for tick computation")
    print(f"{(1000 * avg_gpu_pull):6.2f} ms average time for gpu read operation")
    print(f"{(1000 * t_buffer_write):6.2f} ms time for buffer write operation")
    print("\n")

def _normalize_step_state(expected_state: StateTree, returned_state: StateTree, element: Circuit) -> StateTree:
    """Preserve the compiled state contract across every kernel invocation."""
    expected_keys = set(expected_state)
    returned_keys = set(returned_state)
    if expected_keys != returned_keys:
        missing = sorted(expected_keys - returned_keys)
        extra = sorted(returned_keys - expected_keys)
        raise EngineError(
            f"Kernel for {type(element).__name__}({element.get_path_str()}) returned an invalid "
            f"state tree; missing keys: {missing}, unexpected keys: {extra}."
        )

    normalized_state = {}
    for state_id, expected_value in expected_state.items():
        returned_value = returned_state[state_id]
        if returned_value.shape != expected_value.shape:
            raise EngineError(
                f"Kernel for {type(element).__name__}({element.get_path_str()}) returned "
                f"shape {returned_value.shape} for '{state_id}', expected {expected_value.shape}."
            )
        normalized_state[state_id] = jnp.asarray(returned_value, dtype=expected_value.dtype)
    return normalized_state

def _ensure_compiled(runtime: SimulationRuntime) -> None:
    if runtime is None or not runtime.circuit.is_compiled:
        raise NotCompiledError("Can't run simulation. The circuit is not compiled.")
