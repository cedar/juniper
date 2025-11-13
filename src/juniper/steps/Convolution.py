from .Step import Step
from .. import util
from .. import util_jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial



class Convolution(Step):
    """
    Description
    ---------
    Convolution of incoming step with kernel.

    Parameters
    ---------
    - border_type : float
    - kernel : LateralKernel
    - mode (optional) : str(same)
        - Default = same

    Step Input/Output slots
    ---------
    - in0 : jnp.array()
    - out0 : jnp.array()
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["border_type", "mode", "kernel"]
        super().__init__(name, params, mandatory_params)
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = 1
        self._kernel = self._params["kernel"].get_kernel()

        if "mode" not in self._params.keys():
            self._params["mode"] = "same"

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        
        # Input
        input_mat = input_mats[util.DEFAULT_INPUT_SLOT]
        
        # Computation
        output = jsp.signal.fftconvolve(input_mat, self._kernel, mode=self._params["mode"])

        # Return output
        return {util.DEFAULT_OUTPUT_SLOT: output}
    
