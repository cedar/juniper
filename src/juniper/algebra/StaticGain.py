import logging
from ..core.frontend.Step import Step
from ..core.backend.Exceptions import EngineError
from ..util import util
import jax.numpy as jnp


logger = logging.getLogger(__name__)
# construction of compute kernel
def compute_kernel_factory(factor, params):
    def compute_kernel(input_mats : dict[str,jnp.ndarray], buffer : dict[str,jnp.ndarray], **kwargs) -> dict[str,jnp.ndarray]: 
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        try:
            input = input.astype(params["jdtype"])
        except Exception as e:
            raise EngineError(f"Dtype {input.dtype} of input into step {params.get('name')} can't be converted to {params.get('jdtype')}") from e
        return {util.DEFAULT_OUTPUT_SLOT: input * factor}
    return compute_kernel


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
    def __init__(self, name : str, factor : float):
        params = locals().copy()
        mandatory_params = ["factor"]
        super().__init__(name, params, mandatory_params)
        self.compute_kernel = compute_kernel_factory(self._params["factor"], self._params)
