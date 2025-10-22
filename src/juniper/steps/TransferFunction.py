import jax
from functools import partial
from .Step import Step
from .. import util
from ..Sigmoid import Sigmoid

class TransferFunction(Step):

    def __init__(self, name, params):
        mandatory_params = ["threshold", "beta", "function"]
        super().__init__(name, params, mandatory_params)
        self._trans_func = Sigmoid(self._params["function"]).sigmoid

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = self._trans_func(input, self._params["beta"], self._params["threshold"])
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
