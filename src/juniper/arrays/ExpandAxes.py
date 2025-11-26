import jax
import jax.numpy as jnp
from functools import partial
from ..configurables.Step import Step
from ..util import util

def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = jnp.expand_dims(input, axis=params["axis"])

        for ax, size in zip(params["axis"], params["sizes"]):
            output = jnp.repeat(output, size, axis=ax)
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel

class ExpandAxes(Step):
    """
    Description
    ---------
    Expand incoming step along specified axis.

    Parameters
    ---------
    - axis : tuple(ax0,ax1,...)
    - sizes : tuple(s0,s1,...)
        - sizes per dimension

    Step Input/Output slots
    ---------
    - in0 : jnp.array((Nx,...))
    - out0 : jnp.array((Nx,ax0,ax1,...))
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["axis", "sizes"]
        super().__init__(name, params, mandatory_params)

        self.compute_kernel = compute_kernel_factory(params)
    