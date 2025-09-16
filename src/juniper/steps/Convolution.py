from .Step import Step
from .. import util
from .. import util_jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial



class Convolution(Step):

    def __init__(self, name, params):
        mandatory_params = ["border_type", "mode", "kernel"]
        super().__init__(name, params, mandatory_params)
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = 1
        self._kernel = self._params["kernel"].get_kernel()
        

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        
        # Input
        input_mat = input_mats[util.DEFAULT_INPUT_SLOT]
        
        # Computation
        output = jsp.signal.fftconvolve(input_mat, self._kernel, mode=self._params["mode"])

        # Return output
        return {util.DEFAULT_OUTPUT_SLOT: output}
    
