import jax
from functools import partial
from ..configurables.Step import Step
from ..util import util
from ..configurables.Sigmoid import Sigmoid

class TransferFunction(Step):
    """
    Description
    ---------
    Applies a non-linearity.

    Parameters
    ----------
    - threshold : float
    - beta : float
    - function : str(AbsSigmoid, HeavySideSigmoid, ExpSigmoid, LinearSigmoid, SemiLinearSigmoid, LogarithmicSigmoid)

    Step Input/Output slots
    ----------
    - in0 : jnp.ndarray 
    - out0 : jnp.ndarray 
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["threshold", "beta", "function"]
        super().__init__(name, params, mandatory_params)
        self._trans_func = Sigmoid(self._params["function"]).sigmoid

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = self._trans_func(input, self._params["beta"], self._params["threshold"])
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
