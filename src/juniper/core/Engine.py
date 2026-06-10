from __future__ import annotations

from ..util.util import timer
from ..util import util_jax
import jax
import numpy as np
from functools import partial
from typing import Any
from .Circuit import Circuit
from .Compiler import Compiler
from .RuntimeInfo import CompileInfo
from .RuntimeInfo import RuntimeState
from .RuntimeInfo import StateTree
from .RuntimeInfo import load_permanent_buffers
from .RuntimeInfo import save_permanent_buffers

TimingInfo = dict[str, Any]
Recording = list[list[np.ndarray]]

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

    ########## compilation #################
    def compile(self, circuit : Circuit, warmup : int = 0, print_compile_info : bool = False, load_buffer : bool = False) -> None:
        """Compile a circuit, allocate runtime state, and prepare IO/processes."""
        self.circuit = circuit
        if self.circuit.is_compiled:
            raise Exception(f"Engine::compile(): Circuit is already compiled ({circuit.get_name()})")
        self.compile_info = Compiler().compile(circuit)

        self.kernel_map = self.compile_info.kernel_map
        self.init_state = RuntimeState.from_compile_info(self.compile_info)
        self.state = self.init_state.copy()
        self._init_prng_tree()

        if load_buffer:
            self._load_buffers()
        self._open_connections()

        if warmup > 0:
            self.run_simulation(num_steps=warmup, steps_to_record=[], print_timing=print_compile_info)
            self.reset_state()

    def _init_prng_tree(self) -> None:
        """Build the PRNG tree matching the compiled kernel tree."""
        self.prng_tree, self.prng_slots = util_jax.build_prng_tree(
            self.kernel_map,
            self.compile_info.dynamic_step_paths(),
            self.static_prng_key,
        )

    def _load_buffers(self) -> dict[str, dict[str, Any]]:
        """Load permanent buffers into the already allocated runtime state."""
        if not self.circuit.is_compiled:
            raise RuntimeError("Engine::load_buffers(): Engine must be compiled before loading buffers")
        return load_permanent_buffers(self.compile_info, self.state)

    def clean(self):
        """Resets the Enging into __init__ state for reuse."""
        self.__init__()


    ########## simulation #################
    def run_simulation(
        self,
        num_steps: int,
        steps_to_record: list[str] = [],
        print_timing: bool = True,
        save_buffer: bool = False,
    ) -> tuple[Recording, TimingInfo]:
        """Run the simulation loop for a fixed number of ticks.

        Each tick pushes source data, updates PRNG keys, executes the jitted
        circuit kernel, pulls sinks/recordings, and optionally saves buffers.
        """

        if not self.circuit.is_compiled:
            raise Exception("Engine::run_simulation(): Circuit is not compiled")

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
            t_tick, state_tree = timer(self.tick)(self.state.state_tree, prng_keys)
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

        return history, timing_info

    def reset_state(self) -> None:
        """Reset the runtime state to the post-compilation initial state."""
        self.state = self.init_state.copy()

    @partial(jax.jit, static_argnames=["self"])
    def tick(self, state: StateTree, prng_keys: StateTree) -> StateTree:
        """Execute one compiled circuit tick inside JAX."""
        out = self.circuit.compute_kernel({}, state, **{"prng_keys": prng_keys, "kernel_map": self.kernel_map})
        return out

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

    def _save_buffers(self) -> None:
        """Save buffers marked permanent to the circuit data file."""
        save_permanent_buffers(self.compile_info, self.state)
                    
    def _close_connections(self) -> None:
        """Close all runtime IO endpoints and managed processes."""
        for ref in self.compile_info.runtime_connections():
            ref.element.close()

    def _open_connections(self) -> None:
        """Open all runtime IO endpoints and managed processes."""
        for ref in self.compile_info.runtime_connections():
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
    
    
