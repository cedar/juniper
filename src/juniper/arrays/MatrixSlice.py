import jax
from functools import partial
from ..configurables.Step import Step
from ..util import util


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
        

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = input[tuple(self.slices)]
        return {util.DEFAULT_OUTPUT_SLOT: output}