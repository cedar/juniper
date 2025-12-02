import jax
import jax.numpy as jnp
from functools import partial
from ..configurables.Step import Step
from ..util import util


def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = jnp.flip(input, params["axis"])
        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel

class Flip(Step):
    """
    Description
    ---------
    Flips an input array along specified axes.

    Parameters
    ---------
    - axis: tuple((ax0,ax1,...))
        - List of dimensions that should be flipped.

    Step Input/Output slots
    ---------
    - in0: jnp.array()
    - out0: jnp.array()
    """

    def __init__(self, name : str, params : dict):
        mandatory_params = ["axis"]
        super().__init__(name, params, mandatory_params)
        self.compute_kernel = compute_kernel_factory(params)
