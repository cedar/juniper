import jax
from functools import partial
from ..configurables.Step import Step
from ..util import util

@jax.jit
def _static_gain_compute(input_mats, factor):
    input = input_mats[util.DEFAULT_INPUT_SLOT]
    output = input * factor    
    return {util.DEFAULT_OUTPUT_SLOT: output}

class StaticGain(Step):
    """
    Description
    ---------
    Multiplies input with constant factor.

    Parameters
    ----------
    - factor : float

    Step Input/Output slots
    ----------
    - in0 : jnp.ndarray 
    - out0 : jnp.ndarray 
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["factor"]
        super().__init__(name, params, mandatory_params)

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        factor = self._params["factor"]
        output = input * factor
        return {util.DEFAULT_OUTPUT_SLOT:  output}