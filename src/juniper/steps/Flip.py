import jax
import jax.numpy as jnp
from functools import partial
from .Step import Step
from .. import util


class Flip(Step):
    """
    Flips an input array along specified axes.

    Parameters
    ---------
    - axis: list/tuple
        - List of dimensions that should be flipped.

    Step Computation
    ---------
    - Input: jnp.ndarray 
    - output: jnp.ndarray 
    """

    def __init__(self, name, params):
        mandatory_params = ["axis"]
        super().__init__(name, params, mandatory_params)

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = jnp.flip(input, self._params["axis"])
        return {util.DEFAULT_OUTPUT_SLOT: output}