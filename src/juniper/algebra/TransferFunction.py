import jax
from functools import partial
from ..configurables.Step import Step
from ..util import util
from ..configurables.Sigmoid import Sigmoid

def compute_kernel_factory(params, trans_func):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = trans_func(input, params["beta"], params["threshold"])
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel

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
        self._trans_func = Sigmoid({"sigmoid":self._params["function"]}).sigmoid
        self.compute_kernel = compute_kernel_factory(self._params, self._trans_func)

