import logging
import jax.numpy as jnp
from ..core.frontend.Step import Step
from ..util import util



logger = logging.getLogger(__name__)
def compute_kernel_factory(params, min, max):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = jnp.clip(input, min, max,)
        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel

class Clamp(Step):
    """
    Description
    ---------
    Clamps the values in an array into the range specified by min and max limits.
    
    TODO: Add ability to replace clipped elements with custom values.

    Parameters
    ---------
    - limits: tuple(min,max)

    Step Input/Output slots
    ---------
    - in0 : jnp.array()
    - out0 : jnp.array()
    """
    def __init__(self, name : str, limits : tuple):
        params = locals().copy()
        mandatory_params = ["limits"]
        super().__init__(name, params, mandatory_params)
        
        self._min = self._params["limits"][0]
        self._max = self._params["limits"][1] 
        self.compute_kernel = compute_kernel_factory(self._params, self._min, self._max)
