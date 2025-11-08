import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from functools import partial
from .Step import Step
from .. import util

import jax.debug as jgdb

class Resize(Step):
    """
    Resizes a Matrix to a new shape. Pixel values are interpolated using linear interpolation.
    
    TODO: Interpolation modes are currently limited to linear and nearest neighbor by jax. 

    Parameters
    ---------
    - output_shape: tuple(Nx,Ny,...)

    Step Computation
    ---------
    - Input: jnp.ndarray 
    - output: jnp.ndarray 
        - Matrix resized to have output_shape.
    """

    def __init__(self, name, params):
        mandatory_params = ["output_shape"]
        super().__init__(name, params, mandatory_params)

        self._output_shape = self._params["output_shape"]    

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        input_shape = input.shape

        coords = [jnp.linspace(0, s - 1, n) for s, n in zip(input_shape, self._output_shape)]

        #jgdb.print('dsdc{}', coords)

        grid = jnp.meshgrid(*coords, indexing='ij')

        output = map_coordinates(input, grid, order=1)

        return {util.DEFAULT_OUTPUT_SLOT: output}