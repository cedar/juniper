from ..configurables.Step import Step
from ..util import util
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial



class Convolution(Step):
    """
    Description
    ---------
    Convolution of incoming step with kernel. The kernel can be given directly as a static kernel object (ie. Gaussian) or via an input connection as a dynamic kernel.

    Parameters
    ---------
    - kernel (optional) : LateralKernel
        - A dynamic kernel via input is used if this kernel is unspecified.
    - mode (optional) : str(same)
        - Default = same

    Step Input/Output slots
    ---------
    - in0 : jnp.array()
    - out0 : jnp.array()
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = []
        super().__init__(name, params, mandatory_params)
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = 1
        self._kernel = 0 if "kernel" not in self._params.keys() else self._params["kernel"].get_kernel()
        self._use_dynamic = self._kernel == 0
        if "mode" not in self._params.keys():
            self._params["mode"] = "same"

        self._params["shape"] = (1,) # used for initial warmup to set input

        self.register_input("kernel")

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        
        # Input
        input_mat = input_mats[util.DEFAULT_INPUT_SLOT]

        if self._use_dynamic:
            kernel = input_mats["kernel"]
        else:
            kernel = self._kernel
        
        # Computation
        output = jsp.signal.fftconvolve(input_mat, kernel, mode=self._params["mode"])

        # Return output
        return {util.DEFAULT_OUTPUT_SLOT: output}
    
