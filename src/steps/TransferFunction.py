import jax
from functools import partial
from src.steps.Step import Step
import src.sigmoids as sigmoids
from src import util

class TransferFunction(Step):

    def __init__(self, name, params):
        mandatory_params = ["threshold", "beta", "function"]
        super().__init__(name, params, mandatory_params)
        if self._params["function"] == "AbsSigmoid":
            self._trans_func = sigmoids.AbsSigmoid
        elif self._params["function"] == "ExpSigmoid":
            self._trans_func = sigmoids.ExpSigmoid
        elif self._params["function"] == "HeavySideSigmoid":
            self._trans_func = sigmoids.HeavySideSigmoid
        elif self._params["function"] == "LinearSigmoid":
            self._trans_func = sigmoids.LinearSigmoid
        elif self._params["function"] == "SemiLinearSigmoid":
            self._trans_func = sigmoids.SemiLinearSigmoid
        elif self._params["function"] == "LogarithmicSigmoid":
            self._trans_func = sigmoids.LogarithmicSigmoid
        else:
            raise ValueError(f"Unknown function: {self._params['function']}. Supported functions are: "
                             "AbsSigmoid, ExpSigmoid, HeavySideSigmoid, LinearSigmoid, SemiLinearSigmoid, LogarithmicSigmoid.")

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = self._trans_func(input, self._params["beta"], self._params["threshold"])
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
