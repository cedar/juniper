import jax
from functools import partial
from src.steps.Step import Step
from src import util

@jax.jit
def _static_gain_compute(input_mats, factor):
    input = input_mats[util.DEFAULT_INPUT_SLOT]
    output = input * factor    
    return {util.DEFAULT_OUTPUT_SLOT: output}

class StaticGain(Step):

    def __init__(self, name, params):
        mandatory_params = ["factor"]
        super().__init__(name, params, mandatory_params)

    def compute(self, input_mats, **kwargs):
        return _static_gain_compute(input_mats, self._params["factor"])