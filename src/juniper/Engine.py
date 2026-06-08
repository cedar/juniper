from .util import util
from .util.util import timer
from .util import util_jax
from .util.util_jax import zeros
from .util.util_jax import constant
import jax.numpy as jnp
import jax
import numpy as np
import time
import json
from functools import partial
from .configurables.Circuit import Circuit
from .dft.NeuralField import NeuralField
from .Compiler import Compiler

class Engine:
    def __init__(self):
        self.circuit = None
        self.compile_info = {}
        self.kernel_map = {}
        self.init_state = {}
        self.state = {}

        # prng data
        self.prng_tree = None
        self.prng_slots = []
        self.static_prng_key = util_jax.next_random_key()

    def compile(self, circuit : Circuit, warmup : int = 0, print_compile_info : bool = False, load_buffer : bool = False):
        self.circuit = circuit
        if self.circuit.is_compiled:
            raise Exception(f"Engine::compile(): Circuit is already compiled ({circuit.get_name()})")
        self.compile_info = Compiler().compile(circuit)

        self.kernel_map = self.compile_info["kernel_map"]
        self.init_state = self._init_circuit_state(circuit)
        self.state = self.init_state.copy()
        self._init_prng_tree()

        if load_buffer:
            self.load_buffers()
        self.open_connections()

        if warmup > 0:
            self.run_simulation(num_steps=warmup, steps_to_record=[], print_timing=print_compile_info)
            self.reset_state()

    def run_simulation(self, num_steps, steps_to_record : list[str] = [], print_timing : bool = True, save_buffer : bool = False):
        """
        Parameter
        ---------
        - steps_to_record : list(['step1', 'step1.out0', 'sub_circuit.step1.out0', ...])
        - num_steps : int
        - print_timing (optional) : bool
            - Default = True
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
            t_tick, self.state = timer(self.tick)(self.state, prng_keys)
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
            self.print_timing(timing_info)

        return history, timing_info

    def reset_state(self):
        self.state = self.init_state.copy()

    def _init_step_state(self, element):
        state = {}
        for slot_id, slot in element.output_slot_map.items():
            state[slot_id] = zeros(slot.shape, slot.dtype)
        for buffer_id, buffer in getattr(element, "buffer_map", {}).items():
            if isinstance(element, NeuralField):
                state[buffer_id] = constant(buffer.shape, buffer.dtype, element._params["resting_level"])
            else:
                state[buffer_id] = zeros(buffer.shape, buffer.dtype)

        return state

    def _init_circuit_state(self, circuit):
        state = {}
        for element_name, element in circuit.element_map.items():
            if element is not circuit:
                if isinstance(element, Circuit):
                    sub_state = self._init_circuit_state(element)
                    state[element_name] = sub_state
                else:
                    state[element_name] = self._init_step_state(element)
        for slot_id, slot in circuit.output_slot_map.items():
            state[slot_id] = zeros(slot.shape, slot.dtype)
        return state

    def _state_at_path(self, path):
        state = self.state
        for name in path:
            state = state[name]
        return state

    def _set_state_at_path(self, path, step_state):
        if len(path) == 1:
            self.state[path[0]] = step_state
            return

        state = self.state
        for name in path[:-1]:
            state = state[name]
        state[path[-1]] = step_state

    def _default_output_slot(self, element, step_state):
        if util.DEFAULT_OUTPUT_SLOT in step_state:
            return util.DEFAULT_OUTPUT_SLOT
        if "output" in step_state:
            return "output"
        if len(element.output_slot_map) > 0:
            return next(iter(element.output_slot_map.keys()))
        return util.DEFAULT_OUTPUT_SLOT

    def _init_prng_tree(self):
        dynamic_paths = {
            entry["path"]
            for entry in self.compile_info["dynamic"]
            if not isinstance(entry["element"], Circuit)
        }
        self.prng_tree, self.prng_slots = util_jax.build_prng_tree(self.kernel_map, dynamic_paths, self.static_prng_key)

    @partial(jax.jit, static_argnames=["self"])
    def tick(self, state, prng_keys):
        out = self.circuit.compute_kernel({}, state, **{"prng_keys": prng_keys, "kernel_map": self.kernel_map})
        return out

    def _push_sources(self):
        for entry in self.compile_info["sources"]:
            element = entry["element"]
            if not hasattr(element, "get_data"):
                continue
            data = element.get_data()
            if data is None:
                continue
            step_state = dict(self._state_at_path(entry["path"]))
            slot_id = self._default_output_slot(element, step_state)
            step_state[slot_id] = jnp.array(data, dtype=util_jax.cfg["jdtype"])
            if "output" in step_state:
                step_state["output"] = step_state[slot_id]
            self._set_state_at_path(entry["path"], step_state)

    def _pull_sinks(self):
        for entry in self.compile_info["sinks"]:
            element = entry["element"]
            if not hasattr(element, "set_data"):
                continue
            step_state = self._state_at_path(entry["path"])
            slot_id = self._default_output_slot(element, step_state)
            element.set_data(np.array(step_state[slot_id]))

    def _pull_recordings(self, steps_to_record):
        data = []
        for to_record in steps_to_record:
            data.append(self._record_value(to_record))
        return data


    def _record_value(self, to_record):
        path_str, slot_id = to_record.rsplit(".", 1) if "." in to_record else (to_record, util.DEFAULT_OUTPUT_SLOT)
        return np.array(self._state_at_path(tuple(path_str.split(".")))[slot_id])

    def _save_buffers(self):
        tree = {}
        if self.compile_info is None:
            return

        for path, info in self.compile_info["state_info"].items():
            buffers = {}
            step_state = self._state_at_path(path)
            for buffer_id, buffer in info["buffer_map"].items():
                if getattr(buffer, "permanent", False):
                    buffers[buffer_id] = np.array(step_state[buffer_id]).tolist()
            if len(buffers) > 0:
                tree[".".join(path)] = {"BUFFER": buffers}

        if len(tree) > 0:
            with open(f"{util_jax.cfg['arch_file_path']}.data", "w") as f:
                f.write(json.dumps(tree, indent=4))

    def load_buffers(self):
        if not self.circuit.is_compiled:
            raise RuntimeError("Engine::load_buffers(): Engine must be compiled before loading buffers")

        data_file = util_jax.cfg["arch_file_path"] + self.circuit.get_name() + ".data"
        with open(data_file, "r") as f:
            tree = json.load(f)

        loaded_buffer = {}
        state_info = self.compile_info["state_info"]

        for path_str, step_tree in tree.items():
            path = tuple(path_str.split("."))
            try:
                if path not in state_info:
                    raise Exception(f"No compiled step at path {path_str}")
                if "BUFFER" not in step_tree:
                    raise Exception(f"Invalid buffer format. Expected BUFFER, got {step_tree.keys()}")

                step_state = dict(self._state_at_path(path))
                buffer_info = state_info[path]["buffer_map"]
                loaded_step_buffer = {}

                for buffer_id, buffer_data in step_tree["BUFFER"].items():
                    if buffer_id not in buffer_info:
                        raise Exception(f"Step {path_str} has no buffer '{buffer_id}'")

                    buffer = buffer_info[buffer_id]
                    loaded_array = jnp.array(buffer_data, dtype=buffer.dtype or util_jax.cfg["jdtype"])
                    step_state[buffer_id] = loaded_array
                    loaded_step_buffer[buffer_id] = loaded_array

                self._set_state_at_path(path, step_state)
                loaded_buffer[path_str] = loaded_step_buffer
            except Exception as e:
                print(f"-- Error during Engine::load_buffers('{data_file}') --")
                print(e)
                print("Buffer for step " + path_str + " could not be loaded")

        return loaded_buffer
                    
    def close_connections(self):
        for source_entry in self.compile_info["sources"]:
            source = source_entry["element"] if isinstance(source_entry, dict) else self.get_element(source_entry)
            if hasattr(source, "close"):
                source.close()
        for sink_entry in self.compile_info["sinks"]:
            sink = sink_entry["element"] if isinstance(sink_entry, dict) else self.get_element(sink_entry)
            if hasattr(sink, "close"):
                sink.close()
        for process_entry in self.compile_info["sub_processes"]:
            process = process_entry["element"] if isinstance(process_entry, dict) else process_entry
            if hasattr(process, "close"):
                process.close()

    def open_connections(self):
        for source_entry in self.compile_info["sources"]:
            source = source_entry["element"] if isinstance(source_entry, dict) else self.get_element(source_entry)
            if hasattr(source, "open"):
                source.open()
        for sink_entry in self.compile_info["sinks"]:
            sink = sink_entry["element"] if isinstance(sink_entry, dict) else self.get_element(sink_entry)
            if hasattr(sink, "open"):
                sink.open()
        for process_entry in self.compile_info["sub_processes"]:
            process = process_entry["element"] if isinstance(process_entry, dict) else process_entry
            if hasattr(process, "open"):
                process.open()

    def print_timing(self, timing_info):
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