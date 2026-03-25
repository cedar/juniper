from ..configurables.Step import Step
from ..util import util_jax
from ..util import util
import jax
import jax.numpy as jnp
from functools import partial

def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        # input prod is computed in update_input
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        # Return output
        return {util.DEFAULT_OUTPUT_SLOT: input}
    return compute_kernel


class ComponentMultiply(Step):
    """
    Description
    ---------
    Componentwise multiplication of incoming steps.

    Parameters
    ---------

    Step Input/Output slots
    ---------
    - in0 : jnp.array()
    - out0 : jnp.array()
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = []
        super().__init__(name, params, mandatory_params)
        self.needs_input_connections = True
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = jnp.inf
        self.compute_kernel = compute_kernel_factory(self._params)
    
    def update_input(self, arch, input_slot_shape="shape"):
        # overriding the step.update_input() function to do multiplication instead of summation of inputs
        input_prods = {}
        for input_slot in self.input_slot_names:
            input_prod = None
            incoming_steps = arch.get_incoming_steps(self.get_name() + "." + input_slot)
            if len(incoming_steps) == 0:
                input_prod = util_jax.zeros(self._params[input_slot_shape])
            else:
                for step_slot in incoming_steps:
                    step, slot = step_slot.split(".")
                    # Get output buffer of connected step and add it to the input sum
                    step_output = arch.get_element(step).get_buffer(slot)
                    input_prod = input_prod * step_output if input_prod is not None else step_output
            if input_prod is None:
                raise ValueError(f"Step {self.get_name()} has no valid input sum at slot {input_slot}. This should never happen")
            input_prods[input_slot] = input_prod
        return input_prods

