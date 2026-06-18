from __future__ import annotations
from typing import Any
from .DataClasses import CompileInfo
from .DataClasses import StateTree
from .DataClasses import TimingInfo
from .DataClasses import Recording
from .DataClasses import RecKey

from .Exceptions import NotCompiledError
from .Exceptions import CompilerError

from ...util.util import timer
from ...util import util_jax
import jax
import numpy as np
from functools import partial
from ..frontend.Circuit import Circuit
from .Compiler import Compiler
from .RuntimeState import RuntimeState
from .RuntimeState import load_permanent_buffers
from .RuntimeState import save_permanent_buffers


class Engine:
    """Runtime driver for a compiled circuit.

    Engine owns the simulation loop: inits and keeps runtime state,
    pushes/pulls external IO, calls the jitted tick, and reports timings.

    Pseudocode:
        compile(circuit)
            compile_info = Compiler().compile(circuit)
            state = RuntimeState.from_compile_info(compile_info)
            build PRNG tree
            open runtime connections

        run_simulation(n)
            repeat n times:
                push sources into state
                update PRNG keys
                state = jitted tick(state, keys)
                pull sinks and recordings from state
            optionally save persistent buffers
    """

    def __init__(self) -> None:
        """Create an uncompiled engine with fresh static PRNG data."""
        self.circuit = None
        self.compile_info: CompileInfo | None = None
        self.kernel_map = {}
        self.init_state: RuntimeState | None = None
        self.state: RuntimeState | None = None

        # prng data
        self.prng_tree = None
        self.prng_slots = []
        self.static_prng_key = util_jax.next_random_key()


    def run_simulation(
        self,
        num_steps: int,
        steps_to_record: list[RecKey] = [],
        print_timing: bool = True,
        save_buffer: bool = False,
    ) -> tuple[Recording, TimingInfo]:
        """Run the simulation loop for a fixed number of ticks.

        Each tick pushes source data, updates PRNG keys, executes the jitted
        circuit kernel, pulls sinks/recordings, and optionally saves buffers.
        """

        if not self.circuit.is_compiled:
            raise NotCompiledError(f"Can't run simulation. The circtuit {self.circuit.get_local_circuit_id} is not compiled.")

        history = []
        key_timings = []
        tick_timings = []
        gpu_push_timings = []
        gpu_pull_timings = []
        
        for _ in range(num_steps):
            # Push source data to GPU state.
            t_gpu_push, _ = timer(self._push_sources)()
            gpu_push_timings.append(t_gpu_push)

            # generate pnrg keys
            t_key, prng_keys = timer(util_jax.update_prng_tree)(self.prng_tree, self.prng_slots)
            key_timings.append(t_key)

            # Execute tick function.
            t_tick, state_tree = timer(self._tick)(self.state.state_tree, prng_keys)
            self.state.state_tree = state_tree
            tick_timings.append(t_tick)

            # Pull sink and recorded data from GPU state.
            t_sink_pull, _ = timer(self._pull_sinks)()
            t_recording_pull, data = timer(self._pull_recordings)(steps_to_record)
            history.append(data)
            gpu_pull_timings.append(t_sink_pull+t_recording_pull)
            

        # Save permanent buffers.
        t_buffer_write = 0
        if save_buffer:
            t_buffer_write, _ = timer(self._save_buffers)()

        t_total = np.sum(gpu_push_timings) + np.sum(key_timings) + np.sum(tick_timings) + np.sum(gpu_pull_timings) + t_buffer_write

        timing_info = {"total": t_total, "prng": key_timings, "gpu_push": gpu_push_timings, "gpu_pull": gpu_pull_timings, "tick": tick_timings, "buffer": t_buffer_write, "num_steps": num_steps}

        if print_timing:
            self._print_timing(timing_info)

        return Recording(history, steps_to_record), timing_info

    def compile(self, circuit : Circuit, warmup : int = 0, print_compile_info : bool = False, load_buffer : bool = False) -> None:
        """Compile a circuit, allocate runtime state, and prepare IO/processes."""
        self.circuit = circuit
        if self.circuit.is_compiled:
            raise CompilerError(f"The circuit {circuit.get_local_circuit_id()} is already compiled.")
        t_compile, self.compile_info = timer(Compiler.compile)(circuit)

        self.kernel_map = self.compile_info.kernel_map
        self.init_state = RuntimeState.from_compile_info(self.compile_info)
        self.state = self.init_state.copy()
        self.prng_tree, self.prng_slots = self._init_prng_tree()

        if load_buffer:
            self._load_buffers()
        self._open_connections()

        t_trace, _ = timer(self._tick)(self.state.state_tree, self. prng_tree)

        if warmup > 0:
            t_warmup, _ = timer(self.run_simulation)(num_steps=warmup, steps_to_record=[], print_timing=False)
            self.reset_state()

        if print_compile_info:
            self._print_compile_info({"t_compile": t_compile, "t_trace": t_trace, "t_warmup": t_warmup, "N_static":len(self.compile_info.static), "N_dynamic": len(self.compile_info.dynamic), "N_total":len(self.compile_info.compiled_elements), "N_warmup":warmup})


    def reset_state(self) -> None:
        """Reset the runtime state to the post-compilation initial state."""
        self.state = self.init_state.copy()

    def clean(self):
        """Resets the Enging into __init__ state for reuse."""
        self.__init__()


    @partial(jax.jit, static_argnames=["self"])
    def _tick(self, state: StateTree, prng_keys: StateTree) -> StateTree:
        """Execute one compiled tick inside JAX."""
        new_state = state.copy()

        for element_path, kernel in self.kernel_map.items():
            ref = self.compile_info.compiled_elements[element_path]
            element = ref.element
            
            input = self._gather_element_input(new_state, element)
            new_state[element_path] = kernel(
                input,
                state[element_path],
                **{"prng_key": prng_keys[element_path], "prng_keys": prng_keys[element_path]},
            )

        return new_state

    def _gather_element_input(self, state: StateTree, element: Circuit) -> dict[str, Any]:
        """Build a state input dict for one element."""

        input = {}
        for slot_id, input_slot in element.input_slot_map.items():
            sources = element.parent.connection_map_reversed[input_slot.get_local_circuit_id()]
            input[slot_id] = self._aggregate_slot_values(state, sources, element.input_aggregation)

        if isinstance(element, Circuit):
            for out_slot in element.output_slot_map.values():
                source_slots = element.connection_map_reversed[out_slot.get_local_circuit_id()]
                input[out_slot.get_slot_id()] = self._aggregate_slot_values(state, source_slots, element.input_aggregation)
        
        return input

    def _aggregate_slot_values(self, state: StateTree, slots: list[Any], aggregation: str) -> Any:
        """Resolve source slots in flat state and combine them for one input."""
        if len(slots) == 0:
            return 0

        value = self._read_source_slot(state, slots[0])
        for slot in slots[1:]:
            slot_value = self._read_source_slot(state, slot)
            if aggregation == "product":
                value = value * slot_value
            else:
                value = value + slot_value
        return value

    def _read_source_slot(self, state: StateTree, slot: Any) -> Any:
        """Read a source slot from flat state, following circuit input bridges."""
        source = slot.parent
        slot_id = slot.get_slot_id()

        if isinstance(source, Circuit) and slot_id in source.input_slot_map:
            sources = source.parent.connection_map_reversed[slot.get_local_circuit_id()]
            return self._aggregate_slot_values(state, sources, source.input_aggregation)

        return state[source.get_path()][slot_id]
    

    def _init_prng_tree(self) -> None:
        """Build the PRNG tree matching the compiled kernel tree."""
        return util_jax.build_prng_tree(
            self.kernel_map,
            self.compile_info.dynamic_step_paths(),
            self.static_prng_key,
        )


    def _load_buffers(self) -> dict[str, dict[str, Any]]:
        """Load permanent buffers into the already allocated runtime state."""
        if not self.circuit.is_compiled:
            raise RuntimeError("Engine::load_buffers(): Engine must be compiled before loading buffers")
        return load_permanent_buffers(self.compile_info, self.state)
    
    def _save_buffers(self) -> None:
        """Save buffers marked permanent to the circuit data file."""
        save_permanent_buffers(self.compile_info, self.state)
          

    def _push_sources(self) -> None:
        """Copy CPU-side source data into the runtime state before a tick."""
        for ref in self.compile_info.sources:
            element = ref.element
            data = element.get_data()
            if data is None:
                continue
            self.state.write_source_output(ref, data)

    def _pull_sinks(self) -> None:
        """Copy sink outputs from runtime state back to their Python objects."""
        for ref in self.compile_info.sinks:
            ref.element.set_data(self.state.read_slot(ref))

    def _pull_recordings(self, steps_to_record: list[str]) -> list[np.ndarray]:
        """Read requested recording targets from runtime state."""
        data = []
        for to_record in steps_to_record:
            data.append(self.state.record(self.compile_info, to_record))
        return data

          
    def _close_connections(self) -> None:
        """Close all runtime IO endpoints."""
        for ref in self.compile_info.gather_connections():
            ref.element.close()

    def _open_connections(self) -> None:
        """Open all runtime IO endpoints."""
        for ref in self.compile_info.gather_connections():
            ref.element.open()
    
    ########## utility #################
    def _print_timing(self, timing_info: TimingInfo) -> None:
        t_total = timing_info["total"]
        key_timings = timing_info["prng"]
        gpu_push_timings = timing_info["gpu_push"]
        gpu_pull_timings = timing_info["gpu_pull"]
        tick_timings = timing_info["tick"]
        t_buffer_write = timing_info["buffer"]
        num_steps = timing_info["num_steps"]
        
        ms_per_tick = 1000 * (t_total) / num_steps
        avg_gpu_push = np.mean(gpu_push_timings, axis=0)
        avg_tick = np.mean(tick_timings, axis=0)
        avg_gpu_pull = np.mean(gpu_pull_timings, axis=0)
        avg_prng = np.mean(key_timings, axis=0)

        print(f"{(t_total):6.2f} s total duration [{num_steps} steps]")
        print(f"{ms_per_tick:6.2f} ms / time step")
        print(f"{(1000 * avg_gpu_push):6.2f} ms average time for gpu write operation")
        print(f"{(1000 * avg_prng):6.2f} ms average time for prng key generation")
        print(f"{(1000 * avg_tick):6.2f} ms average time for tick computation")
        print(f"{(1000 * avg_gpu_pull):6.2f} ms average time for gpu read operation")
        print(f"{(1000 * t_buffer_write):6.2f} ms time for buffer write operation")
        print("\n")

    def _print_compile_info(self, timing: TimingInfo) -> None:
        n_static = timing["N_static"]
        n_dynamic = timing["N_dynamic"]
        n_total = timing["N_total"]
        t_compile = timing["t_compile"]
        t_trace = timing["t_trace"]
        t_warmup = timing["t_warmup"]
        N_warmup = timing["N_warmup"]
        print(f"Compiled circuit '{self.circuit.get_local_circuit_id()}' with:")
        print(f"{n_static} static steps,")
        print(f"{n_dynamic} dynamic steps,")
        print(f"making a total of {n_total} steps.")
        print(f"{(t_compile):6.2f} s for compilaton of initial state shapes and dtypes")
        print(f"{(t_trace):6.2f} s for jax tracing of state and compute kernels")
        print(f"{(t_warmup):6.2f} s for warmup run [{N_warmup} steps]")
        print("\n")
