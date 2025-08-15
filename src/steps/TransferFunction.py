import jax
from functools import partial
from src.steps.Step import Step
import src.sigmoids as sigmoids
from src import util
from src.sigmoids import SIGMOID_MAP

class TransferFunction(Step):

    def __init__(self, name, params):
        mandatory_params = ["threshold", "beta", "function"]
        super().__init__(name, params, mandatory_params)
        try:
            self._trans_func = SIGMOID_MAP[self._params["function"]]
        except KeyError:
            raise ValueError(
                f"Unknown function: {self._params['function']}. "
                f"Supported functions are: {', '.join(SIGMOID_MAP)}"
                )

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = self._trans_func(input, self._params["beta"], self._params["threshold"])
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
