import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from functools import partial
from ..configurables.Step import Step
from ..util import util

import jax.debug as jgdb


def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        input_shape = input.shape
        output = jnp.zeros(params["output_shape"])

        #coords = [jnp.linspace(0, s - 1, n) for s, n in zip(input_shape, params["output_shape"])] # border aligned
        coords = [(jnp.arange(n) + 0.5) * (s / n) - 0.5 for s, n in zip(input_shape, params["output_shape"])] # pixel aligned

        grid = jnp.meshgrid(*coords, indexing='ij')

        output += map_coordinates(input, grid, order=params["interpolation"])
        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel


class Resize(Step):
    """
    Description
    ---------
    Resizes a Matrix to a new shape. Pixel values are interpolated using linear interpolation.
    
    TODO: Interpolation modes are currently limited to linear and nearest neighbor by jax. 
    TODO: Currently there is a bug where certain static input will initialize with shape [], leading to an error...

    Parameters
    ---------
    - output_shape : tuple(Nx,Ny,...)
    - interpolation (optional) : int
        - Default = 0
        - 0 -> nearest neighbor
        - 1 -> linear

    Step Input/Output slots
    ---------
    - in0 : jnp.ndarray 
    - out0 : jnp.ndarray 
    """

    def __init__(self, name : str, params : dict):
        mandatory_params = ["output_shape"]
        super().__init__(name, params, mandatory_params)

        if "interpolation" not in self._params.keys():
            self._params["interpolation"] = 0

        self.compute_kernel = compute_kernel_factory(self._params)  

    def reset(self):
        output_shape = self._params["output_shape"]
        self.buffer[util.DEFAULT_OUTPUT_SLOT] = jnp.zeros(output_shape)
        reset_state = {}
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self.buffer[util.DEFAULT_OUTPUT_SLOT]
        return reset_state
    
    def reset_buffer(self, slot_name, slot_shape="shape"):
        output_shape = self._params["output_shape"]
        self.buffer[slot_name] = jnp.zeros(output_shape)