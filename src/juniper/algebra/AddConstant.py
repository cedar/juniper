import jax
from functools import partial
from ..configurables.Step import Step
from ..util import util


# construction of compute kernel
def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = input + params["constant"]
        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel


class AddConstant(Step):    
    """
    Description
    ---------
    Adds a constant.

    Parameters
    ---------
    - constant : float

    Step Input/Output slots
    ---------
    - in0 : jnp.array()
    - out0 : jnp.array()
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["constant"]
        super().__init__(name, params, mandatory_params)
        compute_kernel = compute_kernel_factory(self._params)