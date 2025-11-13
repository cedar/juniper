import jax
import jax.numpy as jnp
from functools import partial
from .Step import Step
from .. import util


class MatrixPadding(Step):
    """
    Description
    ---------
    Padds a Matrix by a number of elements in each dimension.

    TODO: Add ability to pad to specified size.

    Parameters
    ---------
    - border_size : [int | Array | jnp.ndarray])
        - Size of border for each dimension.
        - int or (int,): pad each array dimension with the same number of values both before and after.
        - (before, after): pad each array with before elements before, and after elements after.
        - ((before_1, after_1), (before_2, after_2), ... (before_N, after_N)): specify distinct before and after values for each array dimension.
        - See jax.numpy.pad documentation for reference
    - mode (optional) : str
        - Specifies by what mode the padded values are chosen.
        - See available modes in jax.numpy.pad documentation.
        - Default = "constant"

    Step Input/Output slots
    ---------
    - Input: jnp.ndarray 
    - output: jnp.ndarray 
    """

    def __init__(self, name : str, params : dict):
        mandatory_params = ["border_size"]
        super().__init__(name, params, mandatory_params)

        if "mode" not in self._params:
            self._params["mode"] = "constant"

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = jnp.pad(input, pad_width = self._params["border_size"], mode = self._params["mode"])
        return {util.DEFAULT_OUTPUT_SLOT: output}