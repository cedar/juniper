from .configurables.Step import Slot
from .configurables.Step import Step
from .util import util
import jax.numpy as jnp

def compute_kernel_factory(output_slot_map, input_slot_map, connection_map_reversed):
    def compute_kernel(input, inner_state, **kwargs):
        new_state = inner_state.copy

        # gather sub-inputs fo sub-circuits
        sub_inputs = {}
        for dest, source in connection_map_reversed.items():
            dest_name, dest_slot = dest.split(".")
            source_name, source_slot = source.split(".")
            sub_inputs[dest_name] = {dest_slot: inner_state[source_name][source_slot]}

        # gather external input sums for sub-inputs
        for slot_name, dest in input_slot_map.items():
            dest_name, dest_slot = dest.split(".")
            if dest in connection_map_reversed.keys():
                sub_inputs[dest_name][dest_slot].append(jnp.sum(input[slot_name], axis=0))
            else:
                sub_inputs[dest_name] = {dest_slot: [jnp.sum(input[slot_name], axis=0)]}

        # compute sub-circuits
        for element in connection_map_reversed.keys():
            element_name, _ = element.split(".")
            kernel = kwargs["kernel_map"][element_name]["kernel"]
            sub_kernel = kwargs["kernel_map"][element_name]["sub_kernel"]
            prng_keys = kwargs["prng_keys"]
            new_state[element_name] =  kernel(sub_inputs, inner_state[element_name], **{"prng_keys": prng_keys, "kernel_map":sub_kernel})

        # set output
        out = {}
        for slot_name, out_element in output_slot_map.items():
            out_element_name, out_element_slot = out_element.split(".")
            out[slot_name] = new_state[out_element_name][out_element_slot]

        out["inner_state"] = new_state
        return out # {"inner_state": {"step1":{"buffer":123, "out":123}, "step2"...}, "out_slot1":123, "out_slot2":123,...}

    return compute_kernel

class Circuit:
    _current = None

    def __init__(self, name : str):
        self._name = name
        self.element_map = {}
        self.connection_map_reversed = {}
        self.input_slot_map = {}
        self.max_incoming_connections = {}
        self.output_slot_map = {}
        self.compute_kernel = None

        with self:
            self.build_circuit()

    @classmethod
    def parent_circuit(cls):
        if cls._current is None:
            raise RuntimeError("No active circuit. This should never happen.")
        return cls._current
    
    def __enter__(self, exc_type, exc_val, exc_tb):
        self._previous_circuit = Circuit._current
        Circuit._current = self
        return self

    def __exit__(self):
        Circuit._current = self._previous_circuit

    def __rshift__(self, other):
        parent_circuit = Circuit.parent_circuit()
        if isinstance(other, Step) or isinstance(other, Slot) or isinstance(other, Circuit):
            parent_circuit.connect_to(self.get_name(), other.get_name())
            return other
        elif isinstance(other, str):
            parent_circuit.connect_to(self.get_name(), other)
            other_name = other.split('.', 1)[0]
            return parent_circuit.get_element(other_name)
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")

    def __lshift__(self, other):
        parent_circuit = Circuit.parent_circuit()
        if isinstance(other, Step) or isinstance(other, Slot) or isinstance(other, Circuit):
            parent_circuit.connect_to(other.get_name(), self.get_name())
            return self
        elif isinstance(other, str):
            parent_circuit.connect_to(other, self.get_name())
            return self
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")
    

    def add_element(self, element):
        element_name = element.get_name()
        if element_name in self.element_map.keys():
            raise Exception(f"Circuit::add_element(): Element {element_name} already exists in Circuit.")
        self.element_map[element_name] = element
        for slot_name in element.input_slot_map.keys():
            self.connection_map_reversed[element_name + '.' + slot_name] = []
    
    def get_elements(self):
        return self.element_map
    
    def get_element(self, name):
        if name not in self.element_map:
            raise Exception(f"Architecture::get_element(): Element {name} not found in Architecture")
        return self.element_map[name]
    
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
    
    def connect_to(self, source, dest):
        # source and dest are strings of the form "step_name.slot_name" or "step_name" which will use the first slot (util.DEFAULT_*_SLOT)
        self.check_not_compiled()
        if not isinstance(source, str) or not isinstance(dest, str):
            raise Exception(f"Architecture::connect_to(): source and dest must be strings, but got {type(source)} and {type(dest)}. (TODO: add support for Step and Slot type as argument)")
        # Set default slot if not specified
        if "." not in source:
            source = source + "." + util.DEFAULT_OUTPUT_SLOT
        if "." not in dest:
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

    def get_name(self):
        return self._name

    def register_output_slot(self, slot_name):
        if slot_name in self.output_slot_map.keys():
            raise ValueError(f"Output slot {slot_name} already registered in step {self.get_name()}")
        # Register output slot shortcut
        setattr(self, f"{slot_name}", Slot(self, slot_name))

    def register_input_slot(self, slot_name, max_incoming_connections=1):
        if slot_name in self.input_slot_names:
            raise ValueError(f"Input slot {slot_name} already registered in step {self.get_name()}")
        # Register input slot shortcut
        setattr(self, f"{slot_name}", Slot(self, slot_name))
        # Save max incoming connections
        self._max_incoming_connections[slot_name] = max_incoming_connections
    
    def set_input(self, element_name, slot_name, max_incoming_connections=1):
        self.register_input_slot(slot_name=slot_name, max_incoming_connections=max_incoming_connections)
        # Register internal slot connection
        self.input_slot_map[slot_name] = element_name

    def set_output(self, element_name, slot_name):
        if slot_name in self.output_slot_map.keys():
            raise ValueError(f"Output slot {slot_name} already registered in step {self.get_name()}")
        # Register output slot shortcut
        setattr(self, f"{slot_name}", Slot(self, slot_name))
        # Register slot
        self.output_slot_map[slot_name] = element_name

    def generate_kernel(self):
        self.compute_kernel = compute_kernel_factory(self.output_slot_map, self.input_slot_map, self.connection_map_reversed)

    def build_circuit(self):
        pass