from ..core.frontend.Step import Step
from ..util import util

# construction of compute kernel
def compute_kernel_factory(factor):
    return lambda input_mats, buffer, **kwargs: {util.DEFAULT_OUTPUT_SLOT:  input_mats[util.DEFAULT_INPUT_SLOT] * factor}


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
        self.compute_kernel = compute_kernel_factory(self._params["factor"])
