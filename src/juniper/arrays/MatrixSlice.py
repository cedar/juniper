import jax
from functools import partial
from ..configurables.Step import Step
from ..util import util
import jax.debug as jgdb
import jax.numpy as jnp

def compute_kernel_factory(params, slices):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = input[tuple(slices)]
        return {util.DEFAULT_OUTPUT_SLOT: output}

    return compute_kernel

class MatrixSlice(Step):
    """
    Description
    ---------
    Slices Matrix according to specified slice ranges.

    TODO: Add ability to choose center cutout as a slice mode

    Parameters
    ---------
    - slices: tuple((lower,upper), ...)
        - For each dimension slices specifies the lower and upper indice bounds for slicing. 
        - Absolute indice coordinates are used. So (0,10) will slice the first 10 elements (not 10 in the center).

    Step Input/Output slots
    ---------
    - Input: jnp.ndarray 
    - output: jnp.ndarray 
    """

    def __init__(self, name : str, params : dict):
        mandatory_params = ["slices"]
        super().__init__(name, params, mandatory_params)
        
        self.slices = [slice(self._params["slices"][i][0], self._params["slices"][i][1]) for i in range(len(self._params["slices"]))]
        self.compute_kernel = compute_kernel_factory(self._params, self.slices)

    def reset(self):
        output_shape = ()
        for edg in self._params["slices"]:
            sz = edg[1] - edg[0]
            output_shape += (sz,)
        self.buffer[util.DEFAULT_OUTPUT_SLOT] = jnp.zeros(output_shape)
        reset_state = {}
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self.buffer[util.DEFAULT_OUTPUT_SLOT]
        return reset_state
    
    def reset_buffer(self, slot_name, slot_shape="shape"):
        output_shape = ()
        for edg in self._params["slices"]:
            sz = edg[1] - edg[0]
            output_shape += (sz,)
        self.buffer[slot_name] = jnp.zeros(output_shape)