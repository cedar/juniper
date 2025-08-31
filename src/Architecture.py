from src import util
from src import util_jax
import jax
from functools import partial
import time
import numpy as np
import json
import os

_architecture_singleton = None
def get_arch():
    global _architecture_singleton
    if _architecture_singleton is None:
        _architecture_singleton = Architecture()
    return _architecture_singleton

class Architecture:
    def __init__(self):
        self.element_map = {}
        self.connection_map_reversed = {}
        self.compiled = False
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
        print("\nStatic step compilation graph:\n" + "\n".join([f"{elem[0]:<8} <-- {str(elem[1])}" for elem in compilation_graph_static]) + "\n")

    # If warmup is set to true, the architecture is run once and then reset, effectively precompiling all JIT compilable functions in the architecture (e.g. euler step of fields)
    def compile(self, warmup=True):
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

        ## Warmup
        self.compiled = True
        self.check_compiled()
        if warmup:
            self.tick()
            self.reset_steps()
        
        # Load buffers if any were saved during the last run
        data_file = self.cfg_c["arch_file_path"] + ".data"
        if os.path.exists(data_file):
            print("Loading saved buffers...")
            self.load_buffer(data_file)

    def reset_steps(self):
        self.check_compiled()
        for element in self.element_map.values():
            element.reset()
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
                # Add buffer dict to tree
                tree.update(step_tree)
        with open(f"{self.cfg_c['arch_file_path']}.data", "w") as f:
            f.write(json.dumps(tree, indent=4))

    def load_buffer(self, data_file):
        with open(data_file, "r") as f:
            tree = json.load(f)
            steps = tree.keys()
            for step in steps:
                try:
                    self.get_element(step).load_buffer(tree[step])
                except Exception as e:
                    print(f"-- Error during Architecture::load_buffer('{data_file}') --")
                    print(e)
                    print("Buffer for step " + step + " could not be loaded")
                    

    def run_simulation(self, tick_func, steps_to_plot, num_steps, print_timing=True):
        history = []
        timing_all = []
        start_time = time.time()
        for _ in range(num_steps):

            # Execute tick function
            timing_all.append(tick_func())

            # Save output of steps to plot
            if len(steps_to_plot) > 0:
                data = []
                for to_plot in steps_to_plot:
                    step, slot = to_plot.split(".") if "." in to_plot else [to_plot, util.DEFAULT_OUTPUT_SLOT]
                    data.append(self.get_element(step).get_buffer(slot))
                history.append(data)

        end_time = time.time()
        timing = np.mean(timing_all, axis=0)
        ms_per_tick = 1000 * (end_time - start_time) / num_steps
        if print_timing:
            print(f"{ms_per_tick:6.2f} ms / time step")
            print(f"{(end_time - start_time):6.2f} s total duration\n")
            print(f"{(1000 * timing[0]):6.2f} ms average time for computation of static steps")
            print(f"{(1000 * timing[1]):6.2f} ms average time for dynamic computation")
        
        print("Saving buffers... ", end="", flush=True)
        self.save_buffer()
        print("done")

        return history, ms_per_tick, timing

    def tick(self):
        self.check_compiled()
        start_time = time.time()

        ## -- Update static steps --

        for graph_elem in self.compilation_graph_static_c:
            step_name, incoming_steps = graph_elem
            step = self.get_element(step_name) # TODO put step in graph_elem to avoid this call?

            if step.is_source:
                input_sum = None
            else:
                input_sum = step.update_input(self)
            
            step.buffer = step.compute(input_sum)
        static_update_time = time.time() - start_time

        ## -- Update dynamic steps --

        delta_t = self.cfg_c["delta_t"]
        random_keys = util_jax.next_random_keys(len(self.dynamic_steps_c))
        dynamic_output = []

        # Run compute on dynamic steps
        for i, step in enumerate(self.dynamic_steps_c):
            input_mats = step.update_input(self)
            output = step.compute(input_mats, prng_key=random_keys[i], delta_t=delta_t)
            dynamic_output.append(output)

        # Block output and save to buffers *after* all executions are started to allow jax to parallelize compute calls
        for i, step in enumerate(self.dynamic_steps_c):
            step.post_compute(dynamic_output[i])
        dynamic_update_time = time.time() - (start_time + static_update_time)
        return static_update_time, dynamic_update_time
