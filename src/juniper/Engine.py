
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
from .configurables.Step import Step

def extract_state_from_step(step : Step):
    state = {}
    for buf_name, buffer in step.buffer_map.items():
        state[buf_name] = jnp.zeros((1,)) # has to be infered or given infer_shape(out_slot)

    for out_slot_id, out_slot in step.output_slot_map.items():
        state[out_slot_id] = jnp.zeros((1,)) # has to be infered somehow
    
    return state

def extarct_state_from_circuit(circuit : Circuit):
    state = {}
    for out_slot_id, out_slot in circuit.output_slot_map.items():
        state[out_slot_id] = jnp.zeros((1,)) # has to be infered infer_shape(out_slot)

    state["inner_state"] = {}
    for elment_name, element in circuit.element_map.itmes():
        if isinstance(element, Circuit):
            state["inner_state"] = extarct_state_from_circuit(element)
        if isinstance(element, Step):
            state["inner_state"][elment_name] = extract_state_from_step(element)


class Engine:
    def __init__(self):
        self.compiled = False

        self.state = {}         # state dict of the form {"step_name": {"slot_name": jax_array}}: structure should be fixed. buffer arrays live on gpu. Used as input to jitted tick
        self.graph_info = {}   # static dict of the form {"step_name": {"compute_kernel": step.compute_func, "incoming": {"slot_name": [step_name.slot_name]}, "exposed": bool, "kind": str}}
                                # graph info should be defined at compile time and remain static. Is used to define jitted function
        self.sources = [] # list of stepnames that get a new output from the cpu at every step (so CustomInput etc)
        self.sinks = [] # list of step names which act as sinks, by pushing their output to the cpu at every step (TCPWriter). 
        self.write_buffer_steps = [] # list of step names of which to automatically write the wheight buffer (atm its just the HebbianConnectionSteps and BCM stepps)


    def save_buffer(self): 
        tree = {}
        # Retrieve all static and dynamic steps
        steps = [self.get_element(graph_elem[0]) for graph_elem in self.compilation_graph_static_c]
        steps += self.dynamic_steps_c
        for step in steps:
            step_tree = step.save_buffer()
            if step_tree is not None:
                step_gpu_tree = {}
                # Add buffer dict to tree
                #step_tree = {self._name: {"BUFFER": buffer_dict}}
                step_buffer_dict_to_save = step_tree[step._name]["BUFFER"]
                gpu_buffer_dict = {}
                for key in step_buffer_dict_to_save.keys():
                    gpu_buffer_dict[key] = np.array(self.state[step._name][key]).tolist()
                step_gpu_tree = {step._name: {"BUFFER": gpu_buffer_dict}}
                tree.update(step_gpu_tree)
        with open(f"{self.cfg_c['arch_file_path']}.data", "w") as f:
            f.write(json.dumps(tree, indent=4))

    def load_buffer(self, data_file):
        with open(data_file, "r") as f:
            tree = json.load(f)
            steps = tree.keys()
            loaded_buffer = {}
            for step in steps:
                try:
                    loaded_step_buffer = self.get_element(step).load_buffer(tree[step])
                    loaded_buffer[step] = loaded_step_buffer
                except Exception as e:
                    print(f"-- Error during Architecture::load_buffer('{data_file}') --")
                    print(e)
                    print("Buffer for step " + step + " could not be loaded")
            return loaded_buffer
                    
    def run_simulation(self, tick_func, steps_to_plot, num_steps, print_timing=True, save_buffer=False):
        """
        Parameter
        ---------
        - tick_func : fucntion_object
        - steps_to_plot : list(['step1', 'step1.out0', ...])
        - num_steps : int
        - print_timing (optional) : bool
            - Default = True
        """
        history = []
        tick_timings = []
        gpu_push_timings = []
        gpu_pull_timings = []
        start_time = time.time()
        for _ in range(num_steps):

            # update exposed steps by pushing cpu side outputs mats to gpu
            t_gpu_push = time.time()
            # TODO this implementation for generating prng keys is very slow. On the magnitude of 1 ms. There should be a faster way. Maybe move this into tick func or just use numpy generator
            random_keys = util_jax.next_random_keys(len(self.dynamic_steps_c))
            new_state = dict(self.state)
            for step_name in self.sources:
                step = self.get_element(step_name)
                state_buffer = new_state[step_name]
                class_buffer = step.buffer

                update_source =  step.get_data()
                if update_source is not None:
                    new_output = jnp.array(update_source, dtype=jnp.float32)
                    state_buffer["output"] = new_output
                    class_buffer["output"] = new_output
                new_state[step_name] = state_buffer
            self.state = new_state
            gpu_push_timings.append(time.time()-t_gpu_push)

            # Execute tick function
            t_tick = time.time()
            self.state, _, _ = tick_func(self.state, random_keys)
            tick_timings.append(time.time()-t_tick)

            t_gpu_pull = time.time()
            # pull gpu buffers of steps that are sinks
            for step_name in self.sinks:
                step = self.get_element(step_name)
                step_state = dict(self.state[step_name])
                step.set_data(np.array(step_state[util.DEFAULT_OUTPUT_SLOT]))
            
            # pull gpu buffers for buffers we want to plot
            if len(steps_to_plot) > 0:
                data = []
                for to_plot in steps_to_plot:
                    step, slot = to_plot.split(".") if "." in to_plot else [to_plot, util.DEFAULT_OUTPUT_SLOT]
                    data.append(np.array(self.state[step][slot]))
                history.append(data)
            gpu_pull_timings.append(time.time()-t_gpu_pull)

        t_total = time.time() - start_time

        
        t_buffer_write = time.time()
        # pull gpu buffers for buffers we want to save
        if save_buffer:
            for step_name in self.write_buffer_steps:
                step = self.get_element(step_name)
                for buffer in step.buffer_to_save:
                    step.cpu_buffer[buffer] = np.array(self.state[step_name][buffer])
            self.save_buffer()
        t_buffer_write = (time.time()-t_buffer_write)

        if print_timing:
            ms_per_tick = 1000 * (t_total) / num_steps
            avg_gpu_push = np.mean(gpu_push_timings, axis=0)
            avg_tick = np.mean(tick_timings, axis=0)
            avg_gpu_pull = np.mean(gpu_pull_timings, axis=0)

            print(f"{(t_total):6.2f} s total duration [{num_steps} steps]")
            print(f"{ms_per_tick:6.2f} ms / time step")
            print(f"{(1000 * avg_gpu_push):6.2f} ms average time for gpu write operation")
            print(f"{(1000 * avg_tick):6.2f} ms average time for tick computation")
            print(f"{(1000 * avg_gpu_pull):6.2f} ms average time for gpu read operation")
            if save_buffer: 
                print(f"{(1000 * t_buffer_write):6.2f} ms time for buffer write operation")
            print("\n")

        return history, {"total": t_total, "gpu_push": gpu_push_timings, "gpu_pull": gpu_pull_timings, "tick": tick_timings, "buffer": t_buffer_write}, t_total

    def close_connections(self):
        for source_name in self.sources:
            source = self.get_element(source_name)
            if hasattr(source, "close"):
                source.close()
        for sink_name in self.sinks:
            sink = self.get_element(sink_name)
            if hasattr(sink, "close"):
                sink.close()

    def open_connections(self):
        for source_name in self.sources:
            source = self.get_element(source_name)
            if hasattr(source, "open"):
                source.open()
        for sink_name in self.sinks:
            sink = self.get_element(sink_name)
            if hasattr(sink, "open"):
                sink.open()

    def is_compiled(self):
        return self.compiled

    def check_compiled(self):
        if not self.is_compiled():
            raise util.ArchitectureNotCompiledException
    
    def check_not_compiled(self):
        if self.is_compiled():
            raise util.ArchitectureCompiledException

    def construct_static_compilation_graph(self,):
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

    # If warmup is set to true, the architecture is run once and then reset, effectively precompiling all JIT compilable functions in the architecture (e.g. euler step of fields)
    def compile_old(self, tick_func, warmup=10, print_compile_info=False, load_buffer=False):
        self.check_not_compiled()

        ## Compile
        self.cfg_c = util_jax.cfg
        self.dynamic_steps_c = [element for element in self.element_map.values() if element.is_dynamic]

        # Compute the compilation graph of static steps which determines the order in which they need to be executed.
        self.construct_static_compilation_graph()
        # Precompile static steps
        for graph_elem in self.compilation_graph_static_c:
            step_name, incoming_steps = graph_elem
            step = self.get_element(step_name) # TODO put step in graph_elem to avoid this call?
            step.pre_compile(self)

        # gather all the buffers in a state object and all the inputs/outputs and compute funcs in a graph info obj. This is for the jitted tick only
        for step in self.dynamic_steps_c:
            self.state[step._name] = step.buffer

            incoming = {}
            for slot in step.input_slot_names:
                incoming[slot] = self.get_incoming_steps(step._name + "." + slot)

            self.graph_info[step._name] = {"compute_kernel": step.compute_kernel, "incoming": incoming, "kind": "dynamic", "is_source": step.is_source, "update_input_product":False}
            if step.is_source:
                self.sources.append(step._name)
            if step.is_sink:
                self.sinks.append(step._name)
        
        for graph_elem in self.compilation_graph_static_c:
            step_name, incoming_steps = graph_elem
            step = self.get_element(step_name)
            self.state[step_name] = step.buffer

            incoming = {}
            for slot in step.input_slot_names:
                incoming[slot] = self.get_incoming_steps(step._name + "." + slot)

            self.graph_info[step_name] = {"compute_kernel": step.compute_kernel, "incoming": incoming, "kind": "static", "is_source": step.is_source, "update_input_product":False}
            if step.__class__.__name__=="ComponentMultiply":
                self.graph_info[step._name]["update_input_product"] = True 
            if step.is_source:
                self.sources.append(step._name)
            if step.is_sink:
                self.sinks.append(step._name)
            if len(step.buffer_to_save) != 0:
                self.write_buffer_steps.append(step._name)

        self.open_connections()

        ## Warmup
        self.compiled = True
        self.check_compiled()
        if print_compile_info: 
            print("######## compile run ############") # TODO update logging in juniper in general...
        self.run_simulation(tick_func, steps_to_plot=[], num_steps=warmup, print_timing=print_compile_info)
        #self.reset_steps() # TODO in some unknown steps, reset triggers a new tracing call, thereby resetting the computational graph..
        if print_compile_info: 
            print("############")
        
        # Load buffers if any were saved during the last run
        data_file = self.cfg_c["arch_file_path"] + ".data"
        if os.path.exists(data_file) and load_buffer:
            if print_compile_info: 
                print("Loading saved buffers...")
            loaded_buffer = self.load_buffer(data_file)
            for step_name in loaded_buffer:
                for buffer_name in loaded_buffer[step_name]:
                    self.state[step_name][buffer_name] = loaded_buffer[step_name][buffer_name]


    @partial(jax.jit, static_argnames=["self"])
    def tick_jitted(self, state, rng_keys):

        static_step_names = [key for key, value in self.graph_info.items() if value["kind"]=="static"]
        dynamic_step_names = [key for key, value in self.graph_info.items() if value["kind"]=="dynamic"]

        # 2. Statische Steps
        state = update_static_steps_jax(state, self.graph_info, static_step_names)
        
        # 3. Dynamische Steps
        state = update_dynamic_steps_jax(state, self.graph_info, dynamic_step_names, rng_keys)

        return state, None, None
    
def update_static_steps_jax(state, graph_info, static_step_names):
    new_state = dict(state)
    for step_name in static_step_names:
        step_buffer = state[step_name]
        step_inputs = graph_info[step_name]["incoming"]
        step_compute_kernel = graph_info[step_name]["compute_kernel"]
        input_sums = {}
        for slot, input_steps in step_inputs.items():
            input_sum = None 
            for in_step in input_steps:
                in_step_name, in_step_slot = in_step.split(".")
                #print("CURRENT STEP: ", step_name)
                #print("CURRENT IN: ", in_step)
                #print("CURRENT INPUT SHAPE: ", new_state[in_step_name][in_step_slot].shape)
                if graph_info[step_name]["update_input_product"]:
                    input_sum = (input_sum * new_state[in_step_name][in_step_slot]) if input_sum is not None else new_state[in_step_name][in_step_slot]
                else:
                    input_sum = (input_sum + new_state[in_step_name][in_step_slot]) if input_sum is not None else new_state[in_step_name][in_step_slot]
            #if input_sum is None:
            #    raise ValueError(f"Step {step_name} has no valid input sum at slot {slot}. This should never happen")
            input_sums[slot] = input_sum
        
        #print(input_sums)
        #print(step_buffer)
        new_state[step_name] = step_compute_kernel(input_sums, step_buffer)
    return new_state
        
def update_dynamic_steps_jax(state, graph_info, dynamic_step_names, rng_keys):
    # compute
    new_state = dict(state)
    random_keys = rng_keys
    steps_outputs = {}
    for i, step_name in enumerate(dynamic_step_names):
        step_buffer = state[step_name]
        step_inputs = graph_info[step_name]["incoming"]
        step_compute_kernel = graph_info[step_name]["compute_kernel"]
        input_sums = {}
        for slot, input_steps in step_inputs.items():
            input_sum = None 
            for in_step in input_steps:
                #print("CURRENT STEP: ", step_name)
                #print("CURRENT IN: ", in_step)
                #print("CURRENT SUM: ", input_sum)
                in_step_name, in_step_slot = in_step.split(".")
                
                #jgdb.print("{x}",x=new_state[in_step_name][in_step_slot])
                #print(new_state[in_step_name][in_step_slot])
                input_sum = (input_sum + state[in_step_name][in_step_slot]) if input_sum is not None else state[in_step_name][in_step_slot]
            if len(input_steps) == 0:
                input_sum = 0*state[step_name][util.DEFAULT_OUTPUT_SLOT]
            if input_sum is None:
                raise ValueError(f"Step {step_name} has no valid input sum at slot {slot}. This should never happen. {step_inputs}")
            input_sums[slot] = input_sum
        steps_outputs[step_name] = step_compute_kernel(input_sums, step_buffer, **{"prng_key": random_keys[i]})
    
    # post compute
    for step_name, output in steps_outputs.items():
        #print(output)
        for slot_key in output.keys():
            out_mat = output[slot_key]
            #out_mat.block_until_ready()
            new_state[step_name][slot_key] = out_mat
    
    return new_state


    def load_buffer(self, tree):
        if "BUFFER" not in tree:
            raise Exception(f"Invalid buffer format. Expected BUFFER, got {tree.keys()}")
        buffer_tree = tree["BUFFER"]
        # Iterate through every saved buffer of this step
        step_buffer = {}
        for buffer in buffer_tree.keys():
            # Check if the saved buffer exists in the current step
            if buffer not in self.buffer:
                raise Exception(f"Step {self.get_name()} has no buffer '{buffer}': {list(self.buffer.keys())}")
            buf_str = buffer_tree[buffer]
            #print(type(buf_str))
            buf_array = np.array(buf_str)
            #print(buf_array.shape)
            """metadata, data = buf_str.split("\n")
            # Check metadata format
            metadata = metadata.split(",")
            if not metadata[0] == "Mat":
                raise Exception(f"Invalid buffer format. Expected Mat, got {metadata[0]}")
            # Check datatype
            if not metadata[1] == util_jax.dtype_CV_string():
                raise Exception(f"Datatype of saved buffer ({metadata[1]}) has to match the datatype of the current architecture ({util_jax.dtype_CV_string()})")
            # Fill buffer
            shape = tuple([int(value_str) for value_str in metadata[2:]])
            arr = jnp.array([float(value_str) for value_str in data.split(",")]).reshape(shape)"""

            self.buffer[buffer] = buf_array
            self.cpu_buffer[buffer] = np.array(buf_array)
            step_buffer[buffer] = buf_array
            print(f"loaded buffer for {self.get_name()} with shape {buf_array.shape}")
        return step_buffer

    def save_buffer(self):
        if len(self.buffer_to_save) == 0:
            return None
        buffer_dict = {}
        for buf_name in self.buffer_to_save:
            mat = self.cpu_buffer[buf_name]
            # Save metadata containing datatype and matrix shape
            buf_str = f"Mat,{util_jax.dtype_CV_string()},{','.join([str(size) for size in mat.shape])}" + "\n"
            # Add flattened matrix elements
            buf_str += str(np.asarray(mat).flatten().tolist())[1:-1]
            buffer_dict[buf_name] = buf_str
        # return dict containing all saved buffers
        tree = {self._name: {"BUFFER": buffer_dict}}
        return tree