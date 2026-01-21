from .util import util
from .util import util_jax
import jax
from functools import partial
import time
import numpy as np
import json
import os
import jax.numpy as jnp
import jax.debug as jgdb

_architecture_singleton = None
def get_arch():
    global _architecture_singleton
    if _architecture_singleton is None:
        _architecture_singleton = Architecture()
    return _architecture_singleton

def delete_arch():
    global _architecture_singleton
    _architecture_singleton = None

class Architecture:
    def __init__(self):
        self.element_map = {}
        self.connection_map_reversed = {}
        self.compiled = False

        self.state = {}         # state dict of the form {"step_name": {"slot_name": jax_array}}: structure should be fixed. buffer arrays live on gpu. Used as input to jitted tick
        self.graph_info = {}   # static dict of the form {"step_name": {"compute_kernel": step.compute_func, "incoming": {"slot_name": [step_name.slot_name]}, "exposed": bool, "kind": str}}
                                # graph info should be defined at compile time and remain static. Is used to define jitted function
        self.exposed_steps = [] # list of stepnames that get a new output from the cpu at every step (so CustomInput etc)
        self.write_buffer_steps = [] # list of step names of which to automatically write the whight buffer (atm its just the HebbianConnectionSteps and BCM stepps)


        if not _architecture_singleton is None:
            raise Exception("Do not instantiate this class, use Architecture::get_arch() instead.")

    def is_compiled(self):
        return self.compiled

    @partial(jax.jit, static_argnames=['self'])
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
    def compile(self, tick_func, warmup=10, print_compile_info=False, load_buffer=False):
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
            if step.is_exposed:
                self.exposed_steps.append(step._name)
        
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
            if step.is_exposed:
                self.exposed_steps.append(step._name)
            if len(step.buffer_to_save) != 0:
                self.write_buffer_steps.append(step._name)

        ## Warmup
        self.compiled = True
        self.check_compiled()
        random_keys = util_jax.next_random_keys(len(self.dynamic_steps_c))
        self.run_simulation(tick_func, steps_to_plot=[], num_steps=warmup, print_timing=print_compile_info)
        for i in range(warmup):
            tick_func(self.state, random_keys)
        self.reset_steps()
        
        # Load buffers if any were saved during the last run
        data_file = self.cfg_c["arch_file_path"] + ".data"
        if os.path.exists(data_file) and load_buffer:
            if print_compile_info: print("Loading saved buffers...")
            loaded_buffer = self.load_buffer(data_file)
            for step_name in loaded_buffer:
                for buffer_name in loaded_buffer[step_name]:
                    self.state[step_name][buffer_name] = loaded_buffer[step_name][buffer_name]

    def reset_steps(self):
        self.check_compiled()
        reset_state = {}
        for element in self.element_map.values():
            # reset buffer in individual step
            reset_state[element._name] = element.reset()
            
        self.state = reset_state

        # TODO load saved buffers after reset?

    def add_element(self, element):
        self.check_not_compiled()
        name = element.get_name()
        if name in self.element_map:
            raise Exception(f"Architecture::add_element(): Element {name} already exists in Architecture")
        self.element_map[name] = element

    def register_element_input_slot(self, element_name, slot_name):
        self.connection_map_reversed[element_name + "." + slot_name] = []

    def get_elements(self):
        return self.element_map
    
    def get_element(self, name):
        if name not in self.element_map:
            raise Exception(f"Architecture::get_element(): Element {name} not found in Architecture")
        return self.element_map[name]
    
    # source and dest are strings of the form "step_name.slot_name" or "step_name" which will use the first slot (util.DEFAULT_*_SLOT)
    def connect_to(self, source, dest):
        self.check_not_compiled()
        if not isinstance(source, str) or not isinstance(dest, str):
            raise Exception(f"Architecture::connect_to(): source and dest must be strings, but got {type(source)} and {type(dest)}. (TODO: add support for Step and Slot type as argument)")
        # Set default slot if not specified
        if not "." in source:
            source = source + "." + util.DEFAULT_OUTPUT_SLOT
        if not "." in dest:
            dest = dest + "." + util.DEFAULT_INPUT_SLOT
        
        source_step = source.split(".")[0]
        dest_step = dest.split(".")[0]
        dest_slot = dest.split(".")[1]
        
        for name in [source_step, dest_step]:
            if name not in self.element_map:
                raise Exception(f"Architecture::connect_to(): Element {name} not found in Architecture")
        if source_step == dest_step:
            raise Exception(f"Architecture::connect_to(): Cannot connect element {source} to itself")
        if source in self.connection_map_reversed[dest]:
            raise Exception(f"Architecture::connect_to(): Connection from {source} to {dest} already exists")
        
        if len(self.connection_map_reversed[dest]) >= self.get_element(dest_step).get_max_incoming_connections(dest_slot):
            raise Exception(f"Architecture::connect_to(): Element {dest} already has {self.get_element(dest_step).get_max_incoming_connections(dest_slot)} incoming connection(s)")

        self.connection_map_reversed[dest].append(source)
    
    # dest is a string of the form "step_name.slot_name" or "step_name" which will return all incoming steps to that step
    def get_incoming_steps(self, dest):
        incoming_steps = []
        if "." in dest:
            # If slot is specified, return all incoming steps to that slot
            incoming_steps = self.connection_map_reversed[dest]
        else:
            # If no slot is specified, return all incoming steps to all slots
            all_slots = self.get_element(dest).input_slot_names
            for slot in all_slots:
                incoming_steps += self.connection_map_reversed[dest + "." + slot]
        return incoming_steps

    def save_buffer(self): 
        tree = {}
        # Retrieve all static and dynamic steps
        steps = [self.get_element(graph_elem[0]) for graph_elem in self.compilation_graph_static_c]
        steps += self.dynamic_steps_c
        for step in steps:
            step_tree = step.save_buffer()
            if not step_tree is None:
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

    def set_arch_name(self, name : str):
        util_jax.get_config()["arch_file_path"] = name

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
        timing_all = []
        start_time = time.time()
        for _ in range(num_steps):

            random_keys = util_jax.next_random_keys(len(self.dynamic_steps_c))

            # update exposed steps by pushing cpu side outputs mats to gpu
            new_state = dict(self.state)
            for step_name in self.exposed_steps:
                step = self.get_element(step_name)
                state_buffer = new_state[step_name]
                class_buffer = step.buffer
                new_output =  step.output
                state_buffer["output"] = new_output
                class_buffer["output"] = new_output
                new_state[step_name] = state_buffer

            self.state = new_state

            # Execute tick function
            tick_start = time.time()
            self.state, _, _ = tick_func(self.state, random_keys)
            timing_all.append(time.time()-tick_start)
            # update output buffers of exposed steps
            # Save output of steps to plot
            
            if len(steps_to_plot) > 0:
                data = []
                for to_plot in steps_to_plot:
                    step, slot = to_plot.split(".") if "." in to_plot else [to_plot, util.DEFAULT_OUTPUT_SLOT]
                    data.append(np.array(self.state[step][slot]))
                history.append(data)
            
            # pull gpu buffers for buffers we want to save
            for step_name in self.write_buffer_steps:
                step = self.get_element(step_name)
                for buffer in step.buffer_to_save:
                    step.cpu_buffer[buffer] = np.array(self.state[step_name][buffer])
                


        end_time = time.time()
        timing = np.mean(timing_all, axis=0)
        ms_per_tick = 1000 * (end_time - start_time) / num_steps
        if print_timing:
            print(f"{ms_per_tick:6.2f} ms / time step")
            print(f"{(end_time - start_time):6.2f} s total duration\n")
            print(f"{(1000 * timing):6.2f} ms average time for computation")
        
        if save_buffer:
            self.save_buffer()

        return history, ms_per_tick, timing

    def tick(self, state, rng_keys):
        self.check_compiled()
        start_time = time.time()
        new_state = {}

        ## -- Update static steps --

        for graph_elem in self.compilation_graph_static_c:
            step_name, incoming_steps = graph_elem
            step = self.get_element(step_name) # TODO put step in graph_elem to avoid this call?

            if step.is_source:
                input_sum = None
            else:
                input_sum = step.update_input(self)
            
            step.buffer = step.compute(input_sum, step.buffer)
            if step.is_exposed:
                # copy over output state from cpu for steps with externally set output
                step.buffer["output"] =  jax.device_put(step.output, device=jax.devices("gpu")[0])
            new_state[step_name] = step.buffer
        static_update_time = time.time() - start_time

        ## -- Update dynamic steps --

        delta_t = self.cfg_c["delta_t"]
        random_keys = util_jax.next_random_keys(len(self.dynamic_steps_c))
        dynamic_output = []

        # Run compute on dynamic steps
        for i, step in enumerate(self.dynamic_steps_c):
            input_mats = step.update_input(self)
            output = step.compute(input_mats, step.buffer, prng_key=random_keys[i], delta_t=delta_t)
            dynamic_output.append(output)

        # Block output and save to buffers *after* all executions are started to allow jax to parallelize compute calls
        for i, step in enumerate(self.dynamic_steps_c):
            step.post_compute(dynamic_output[i])
            new_state[step._name] = step.buffer
        dynamic_update_time = time.time() - (start_time + static_update_time)
        
        return new_state, static_update_time, dynamic_update_time

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

