import jax
import jax.numpy as jnp
from functools import partial
from .Step import Step
from .. import util


class Clamp(Step):
    """
    Clamps the values in an array into the range specified by min and max limits.
    
    TODO: Add ability to replace clipped elements with custom values.

    Parameters
    ---------
    - limits: tuple(min,max)

    Step Computation
    ---------
    - Input: jnp.ndarray 
    - output: jnp.ndarray 
    """

    def __init__(self, name, params):
        mandatory_params = ["limits"]
        super().__init__(name, params, mandatory_params)
        
        self._min = self._params["limits"][0]
        self._max = self._params["limits"][1] 

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = jnp.clip(input, self._min, self._max,)
        return {util.DEFAULT_OUTPUT_SLOT: output}