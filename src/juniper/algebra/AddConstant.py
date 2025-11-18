import jax
from functools import partial
from ..configurables.Step import Step
from ..util import util


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

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = input + self._params["constant"]
        return {util.DEFAULT_OUTPUT_SLOT: output}