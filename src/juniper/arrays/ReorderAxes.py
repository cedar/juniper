import jax
import jax.numpy as jnp
from functools import partial
from ..configurables.Step import Step
from ..util import util

def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = jnp.transpose(input, axes=params["order"])
        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel

class ReorderAxes(Step):
    """
    Description
    ---------
    Permutation of the axis of the incoming step.

    Parameters
    ----------
    - order : tuple(axi,axj,...)

    Step Input/Output slots
    -----------
    - in0 : jnp.ndarray
    - out0 : jnp.ndarray
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["order"]
        super().__init__(name, params, mandatory_params)
        self.compute_kernel = compute_kernel_factory(self._params)
        

    