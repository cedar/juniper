import jax
from functools import partial
from .Step import Step
from .. import util

@jax.jit
def _static_gain_compute(input_mats, factor):
    input = input_mats[util.DEFAULT_INPUT_SLOT]
    output = input * factor    
    return {util.DEFAULT_OUTPUT_SLOT: output}

class StaticGain(Step):

    def __init__(self, name, params):
        mandatory_params = ["factor"]
        super().__init__(name, params, mandatory_params)

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        factor = self._params["factor"]
        output = input * factor
        return {util.DEFAULT_OUTPUT_SLOT:  output}