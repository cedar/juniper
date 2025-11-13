import jax
import jax.numpy as jnp
from functools import partial
from .Step import Step
from .. import util


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

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = jnp.flip(input, self._params["axis"])
        return {util.DEFAULT_OUTPUT_SLOT: output}