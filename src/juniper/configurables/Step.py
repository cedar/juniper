from ..util import util_jax
from ..util import util
from .Configurable import Configurable
from ..Architecture import get_arch
import numpy as np
import jax.numpy as jnp
import jax

class Slot():
    def __init__(self, step, slot_name):
        self._step = step
        self.name = step.get_name() + "." + slot_name

    def __rshift__(self, other):
        if isinstance(other, Step):
            get_arch().connect_to(self.name, other.get_name())
            return other
        elif isinstance(other, str):
            get_arch().connect_to(self.name, other)
            other_name = other.split('.', 1)[0]
            return get_arch().get_element(other_name)
        elif isinstance(other, Slot):
            get_arch().connect_to(self.name, other.name)
            return other
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")

    def __lshift__(self, other):
        if isinstance(other, Step):
            get_arch().connect_to(other.get_name(), self.name)
        elif isinstance(other, str):
            get_arch().connect_to(other, self.name)
        elif isinstance(other, Slot):
            get_arch().connect_to(other.name, self.name)
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")

class Step(Configurable):
    def __init__(self, name : str, params : dict, mandatory_params : list, is_dynamic : bool = False):
        super().__init__(params, mandatory_params)
        self.is_exposed = False
        self.compute_kernel = None

        self._name = name
        if "." in name:
            raise ValueError(f"Step names cannot contain dots. ({name})")
        self.is_dynamic = is_dynamic
        if is_dynamic and not "shape" in params:
            raise ValueError(f"Dynamic steps require a shape parameter. ({name})")
        self._max_incoming_connections = {}
        self.needs_input_connections = True
        self.is_source = False
        self.buffer = {} # Stores matrices of internal and output buffers
        self.buffer_to_save = []
        self.output_slot_names = []
        self.input_slot_names = []
        get_arch().add_element(self)

        self.register_input(util.DEFAULT_INPUT_SLOT)
        self.register_output(util.DEFAULT_OUTPUT_SLOT)


    def compute(self, input_mats, buffer, **kwargs):
        raise NotImplementedError("Please override compute() in subclasses of Step.")

    def get_max_incoming_connections(self, slot_name):
        return self._max_incoming_connections[slot_name]

    def get_name(self):
        return self._name
    
    def reset(self):
        reset_state = {}
        for name in self.output_slot_names:
            self.reset_buffer(name)
            reset_state[name] = self.buffer[name]#jax.device_put(self.buffer[name], device= jax.devices("gpu")[0])
        return reset_state
        
    def reset_buffer(self, slot_name, slot_shape="shape"):
        if not self.is_dynamic:
            self.buffer[slot_name] = jnp.array([])
        else:
            self.buffer[slot_name] = util_jax.zeros(self._params[slot_shape])

    def pre_compile(self, arch):
        if not self.is_source:
            incoming_steps = arch.get_incoming_steps(self.get_name())
            if len(incoming_steps) == 0 and self.needs_input_connections:
                raise ValueError(f"Step {self.get_name()} has no incoming connection")
    
    def get_buffer(self, buf_name):
        if buf_name in self.buffer:
            return self.buffer[buf_name]
        else:
            raise Exception(f"Buffer {buf_name} not found ({self.get_name()})")
        
    def __rshift__(self, other):
        if isinstance(other, Step):
            get_arch().connect_to(self.get_name(), other.get_name())
            return other
        elif isinstance(other, str):
            get_arch().connect_to(self.get_name(), other)
            other_name = other.split('.', 1)[0]
            return get_arch().get_element(other_name)
        elif isinstance(other, Slot):
            get_arch().connect_to(self.get_name(), other.name)
            return other
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")

    def __lshift__(self, other):
        if isinstance(other, Step):
            get_arch().connect_to(other.get_name(), self.get_name())
        elif isinstance(other, str):
            get_arch().connect_to(other, self.get_name())
        elif isinstance(other, Slot):
            get_arch().connect_to(other.name, self.get_name())
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")
    
    def register_output(self, slot_name):
        # Initialize output buffer
        self.register_buffer(slot_name)
        # Register output slot shortcut
        setattr(self, f"o{len(self.output_slot_names)}", Slot(self, slot_name))
        # Save output slot name
        if slot_name in self.output_slot_names:
            raise ValueError(f"Output slot {slot_name} already registered in step {self.get_name()}")
        self.output_slot_names.append(slot_name)

    def register_input(self, slot_name, max_incoming_connections=1):
        # Register input slot shortcut
        setattr(self, f"i{len(self.output_slot_names)}", Slot(self, slot_name))
        # Register input slot
        if slot_name in self.input_slot_names:
            raise ValueError(f"Input slot {slot_name} already registered in step {self.get_name()}")
        self.input_slot_names.append(slot_name)
        # Save max incoming connections
        self._max_incoming_connections[slot_name] = max_incoming_connections
        get_arch().register_element_input_slot(self.get_name(), slot_name)
    
    def register_buffer(self, buf_name, slot_shape="shape", save=False):
        if buf_name in self.buffer:
            raise ValueError(f"Buffer {buf_name} already registered ({self.get_name()})")
        self.reset_buffer(buf_name, slot_shape)
        if save:
            self.buffer_to_save.append(buf_name)

    def load_buffer(self, tree):
        if not "BUFFER" in tree:
            raise Exception(f"Invalid buffer format. Expected BUFFER, got {tree.keys()}")
        buffer_tree = tree["BUFFER"]
        # Iterate through every saved buffer of this step
        for buffer in buffer_tree.keys():
            # Check if the saved buffer exists in the current step
            if not buffer in self.buffer:
                raise Exception(f"Step {self.get_name()} has no buffer '{buffer}': {list(self.buffer.keys())}")
            buf_str = buffer_tree[buffer]
            metadata, data = buf_str.split("\n")
            # Check metadata format
            metadata = metadata.split(",")
            if not metadata[0] == "Mat":
                raise Exception(f"Invalid buffer format. Expected Mat, got {metadata[0]}")
            # Check datatype
            if not metadata[1] == util_jax.dtype_CV_string():
                raise Exception(f"Datatype of saved buffer ({metadata[1]}) has to match the datatype of the current architecture ({util_jax.dtype_CV_string()})")
            # Fill buffer
            shape = tuple([int(value_str) for value_str in metadata[2:]])
            arr = jnp.array([float(value_str) for value_str in data.split(",")]).reshape(shape)
            self.buffer[buffer] = arr

    def save_buffer(self):
        if len(self.buffer_to_save) == 0:
            return None
        buffer_dict = {}
        for buf_name in self.buffer_to_save:
            mat = self.buffer[buf_name]
            # Save metadata containing datatype and matrix shape
            buf_str = f"Mat,{util_jax.dtype_CV_string()},{','.join([str(size) for size in mat.shape])}" + "\n"
            # Add flattened matrix elements
            buf_str += str(np.asarray(mat).flatten().tolist())[1:-1]
            buffer_dict[buf_name] = buf_str
        # return dict containing all saved buffers
        tree = {self._name: {"BUFFER": buffer_dict}}
        return tree

    def update_input(self, arch, input_slot_shape="shape"):
        input_sums = {}
        for input_slot in self.input_slot_names:
            input_sum = None
            incoming_steps = arch.get_incoming_steps(self.get_name() + "." + input_slot)
            if len(incoming_steps) == 0:
                input_sum = util_jax.zeros(self._params[input_slot_shape])
            else:
                for step_slot in incoming_steps:
                    step, slot = step_slot.split(".")
                    # Get output buffer of connected step and add it to the input sum
                    step_output = arch.get_element(step).get_buffer(slot)
                    input_sum = input_sum + step_output if input_sum is not None else step_output
            if input_sum is None:
                raise ValueError(f"Step {self.get_name()} has no valid input sum at slot {input_slot}. This should never happen")
            input_sums[input_slot] = input_sum
        return input_sums
    
    # Only gets called in dynamic steps
    def post_compute(self, output_matrices):
        # Output of compute is now ready, save it to output buffer
        for key in output_matrices.keys():
            output = output_matrices[key]
            output.block_until_ready()
            self.buffer[key] = output
