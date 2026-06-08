
from .util import util
from .util import util_jax
import jax.numpy as jnp
import jax
import numpy as np
import time
import json
import os
from functools import partial
from .configurables.Circuit import Circuit
from .dft.NeuralField import NeuralField

class Engine:
    def __init__(self, circuit):
        self.compiled = False
        self.circuit = circuit
        self.compile_info = None
        self.kernel_map = {}

    def compile(self, circuit : Circuit, input_slots=None, warmup=0, print_compile_info=False, load_buffer=False):
        self.check_not_compiled()
        circuit.generate_kernel()
        circuit.compile_state(input_slots or {})

        if not circuit.is_compiled:
            print("Not compiled Elements: ")
            for element in self.circuit.element_map.values():
                if not element.is_compiled:
                    print(element.get_name())
            raise RuntimeError(f"Engine::compile(): Circuit {circuit.get_name()} did not compile successfully")

        self.circuit = circuit
        self.compile_info = circuit.compile_info
        self.kernel_map = self.compile_info["kernel_map"]
        self.prng_tree = None
        self.prng_slots = []
        self.static_prng_key = util_jax.next_random_key()
        self._init_prng_tree()
        self.state = self._init_circuit_state(circuit)
        self.sources = self.compile_info["sources"]
        self.sinks = self.compile_info["sinks"]
        self.sub_processes = self.compile_info["sub_processes"]
        self.compiled = True

        data_file = util_jax.cfg["arch_file_path"] + circuit.get_name() + ".data"
        if os.path.exists(data_file) and load_buffer:
            if print_compile_info:
                print("Loading saved buffers...")
            self.load_buffers(data_file)

        self.open_connections()
        if warmup > 0:
            self.run_simulation(num_steps=warmup, steps_to_record=[], print_timing=print_compile_info)

    def _zeros(self, shape, dtype=None):
        return jnp.zeros(shape, dtype=dtype or util_jax.cfg["jdtype"])
    
    def _ones(self, shape, dtype=None):
        return jnp.ones(shape, dtype=dtype or util_jax.cfg["jdtype"])
    
    def _constant(self, shape, constant, dtype=None):
        return self._ones(shape, dtype=dtype) * constant

    def _init_step_state(self, element):
        state = {}
        for slot_id, slot in element.output_slot_map.items():
            state[slot_id] = self._zeros(slot.shape, slot.dtype)
        for buffer_id, buffer in getattr(element, "buffer_map", {}).items():
            if isinstance(element, NeuralField):
                state[buffer_id] = self._constant(buffer.shape, element._params["resting_level"], buffer.dtype)
            else:
                state[buffer_id] = self._zeros(buffer.shape, buffer.dtype)

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
            state[slot_id] = self._zeros(slot.shape, slot.dtype)
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
        self.prng_tree = self._build_prng_tree(self.kernel_map, dynamic_paths)

    def _build_prng_tree(self, kernel_map, dynamic_paths, path=()):
        tree = {}
        for element_name, kernel_info in kernel_map.items():
            element_path = path + (element_name,)
            if kernel_info["sub_kernel"] is None:
                tree[element_name] = self.static_prng_key
                if element_path in dynamic_paths:
                    self.prng_slots.append((tree, element_name))
            else:
                tree[element_name] = self._build_prng_tree(kernel_info["sub_kernel"], dynamic_paths, element_path)
        return tree

    def _next_prng_tree(self):
        if len(self.prng_slots) == 0:
            return self.prng_tree
        keys = util_jax.next_random_keys(len(self.prng_slots))
        for key, (tree, element_name) in zip(keys, self.prng_slots):
            tree[element_name] = key
        return self.prng_tree

    @partial(jax.jit, static_argnames=["self"])
    def tick(self, state, prng_keys):
        out = self.circuit.compute_kernel({}, state, **{"prng_keys": prng_keys, "kernel_map": self.kernel_map})
        return out

    def _push_sources(self):
        for entry in self.sources:
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
        for entry in self.sinks:
            element = entry["element"]
            if not hasattr(element, "set_data"):
                continue
            step_state = self._state_at_path(entry["path"])
            slot_id = self._default_output_slot(element, step_state)
            element.set_data(np.array(step_state[slot_id]))

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

    def load_buffers(self, data_file):
        if self.compile_info is None:
            raise RuntimeError("Engine::load_buffers(): Engine must be compiled before loading buffers")

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
                    
    def run_simulation(self, steps_to_record=[], num_steps=None, print_timing=True, save_buffer=False):
        """
        Parameter
        ---------
        - steps_to_record : list(['step1', 'step1.out0', 'sub_circuit.step1.out0', ...])
        - num_steps : int
        - print_timing (optional) : bool
            - Default = True
        """
        self.check_compiled()
        if num_steps is None:
            raise ValueError("Engine::run_simulation(): num_steps must be specified")

        history = []
        key_timings = []
        tick_timings = []
        gpu_push_timings = []
        gpu_pull_timings = []
        start_time = time.time()
        for _ in range(num_steps):

            # Push source data to GPU state.
            t_gpu_push = time.time()
            self._push_sources()
            gpu_push_timings.append(time.time()-t_gpu_push)

            # generate pnrg keys
            t_key = time.time()
            prng_keys = self._next_prng_tree()
            key_timings.append(time.time()-t_key)


            # Execute tick function.
            t_tick = time.time()
            self.state = self.tick(self.state, prng_keys)
            tick_timings.append(time.time()-t_tick)

            # Pull sink and recorded data from GPU state.
            t_gpu_pull = time.time()
            self._pull_sinks()
            
            if len(steps_to_record) > 0:
                data = []
                for to_record in steps_to_record:
                    data.append(self._record_value(to_record))
                history.append(data)
            gpu_pull_timings.append(time.time()-t_gpu_pull)

        t_total = time.time() - start_time

        
        t_buffer_write = time.time()
        # Save permanent buffers.
        if save_buffer:
            self._save_buffers()
        t_buffer_write = (time.time()-t_buffer_write)

        if print_timing:
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
            if save_buffer: 
                print(f"{(1000 * t_buffer_write):6.2f} ms time for buffer write operation")
            print("\n")

        return history, {"total": t_total, "prng": key_timings, "gpu_push": gpu_push_timings, "gpu_pull": gpu_pull_timings, "tick": tick_timings, "buffer": t_buffer_write}, t_total

    def close_connections(self):
        for source_entry in self.sources:
            source = source_entry["element"] if isinstance(source_entry, dict) else self.get_element(source_entry)
            if hasattr(source, "close"):
                source.close()
        for sink_entry in self.sinks:
            sink = sink_entry["element"] if isinstance(sink_entry, dict) else self.get_element(sink_entry)
            if hasattr(sink, "close"):
                sink.close()
        for process_entry in getattr(self, "sub_processes", []):
            process = process_entry["element"] if isinstance(process_entry, dict) else process_entry
            if hasattr(process, "close"):
                process.close()

    def open_connections(self):
        for source_entry in self.sources:
            source = source_entry["element"] if isinstance(source_entry, dict) else self.get_element(source_entry)
            if hasattr(source, "open"):
                source.open()
        for sink_entry in self.sinks:
            sink = sink_entry["element"] if isinstance(sink_entry, dict) else self.get_element(sink_entry)
            if hasattr(sink, "open"):
                sink.open()
        for process_entry in getattr(self, "sub_processes", []):
            process = process_entry["element"] if isinstance(process_entry, dict) else process_entry
            if hasattr(process, "open"):
                process.open()

    def is_compiled(self):
        return self.compiled

    def check_compiled(self):
        if not self.is_compiled():
            raise util.ArchitectureNotCompiledException
    
    def check_not_compiled(self):
        if self.is_compiled():
            raise util.ArchitectureCompiledException

"""    def construct_static_compilation_graph(self,):
        self.check_not_compiled()
        compiled_steps = []
        compilation_graph_static = []
        # iterate through all dynamic steps and create a subgraph of the tree of incoming static steps to this dynamic step.
        for dynamic_step in self.dynamic_steps_c:
            subgraph = []
            incoming_steps = [step.split(".")[0] for step in self.get_incoming_steps(dynamic_step.get_name())]
            if len(incoming_steps) == 0:
                continue
            bfs_queue = incoming_steps
            # Do a backwards BFS to find the subtree
            while len(bfs_queue) > 0:
                current = bfs_queue.pop(0)
                if current in compiled_steps or self.element_map[current].is_dynamic:
                    continue
                compiled_steps.append(current)
                incoming_steps = [step.split(".")[0] for step in self.get_incoming_steps(current)]
                subgraph.append([current, incoming_steps]) # TODO remove incoming_steps?
                bfs_queue += incoming_steps
            subgraph = subgraph[::-1]
            compilation_graph_static += subgraph
        self.compilation_graph_static_c = compilation_graph_static
        #print("\nStatic step compilation graph:\n" + "\n".join([f"{elem[0]:<8} <-- {str(elem[1])}" for elem in compilation_graph_static]) + "\n")
"""
