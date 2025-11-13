import jax
import jax.numpy as jnp
from functools import partial
from .Step import Step
from .. import util

NORM_ORDER_MAP = {
    "InfinityNorm": jnp.inf,
    "L1Norm": 1,
    "L2Norm": 2,
}

class Normalization(Step):
    """
    Description
    ---------
    Normalizes incoming step using specified norm function.

    Parameters
    ---------    
    - function : str(InfinityNorm, L1Norm, L2Norm)

    Step Input/Output slots
    ---------
    - Input: jnp.ndarray
    - output: jnp.ndarray
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["function"]
        super().__init__(name, params, mandatory_params)
        try:
            self._ord = NORM_ORDER_MAP[self._params["function"]]
        except KeyError:
            raise ValueError(
                f"Unknown function: {self._params['function']}. "
                f"Supported functions are: {', '.join(NORM_ORDER_MAP)}")

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        epsilon = 1e-8 # to avoid division by zero
        output = input / (epsilon + jnp.linalg.norm(input, ord=self._ord))
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
    