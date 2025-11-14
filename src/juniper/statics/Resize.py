import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from functools import partial
from ..configurables.Step import Step
from ..util import util

import jax.debug as jgdb

class Resize(Step):
    """
    Description
    ---------
    Resizes a Matrix to a new shape. Pixel values are interpolated using linear interpolation.
    
    TODO: Interpolation modes are currently limited to linear and nearest neighbor by jax. 

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
            self._params["keys"] = 0

        self._output_shape = self._params["output_shape"]    

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        input_shape = input.shape

        coords = [jnp.linspace(0, s - 1, n) for s, n in zip(input_shape, self._output_shape)]

        grid = jnp.meshgrid(*coords, indexing='ij')

        output = map_coordinates(input, grid, order=self._params["interpolation"])

        return {util.DEFAULT_OUTPUT_SLOT: output}