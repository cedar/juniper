import jax
import jax.numpy as jnp
from functools import partial
from .Step import Step
from .. import util


class AddConstant(Step):    
    """
    Adds a constant.

    Parameters
    ---------
    - constant: float

    Step Computation
    ---------
    - Input: jnp.ndarray 
    - output: jnp.ndarray 
    """

    def __init__(self, name, params):
        mandatory_params = ["constant"]
        super().__init__(name, params, mandatory_params)

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = input + self._params["constant"]
        return {util.DEFAULT_OUTPUT_SLOT: output}