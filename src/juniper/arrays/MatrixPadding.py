import logging
import jax.numpy as jnp
from ..core.frontend.Step import Step
from ..util import util


logger = logging.getLogger(__name__)
def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        output = jnp.pad(input, pad_width = params["border_size"], mode = params["mode"])
        return {util.DEFAULT_OUTPUT_SLOT: output}

    return compute_kernel

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

    _mode = "constant"
    def __init__(self, name : str, border_size, mode : str = _mode):
        params = locals().copy()
        mandatory_params = ["border_size"]
        super().__init__(name, params, mandatory_params)

        self.compute_kernel = compute_kernel_factory(self._params)

    def infer_output_shapes(self, input_specs):
        if util.DEFAULT_INPUT_SLOT not in input_specs:
            return {}
        shape = tuple(input_specs[util.DEFAULT_INPUT_SLOT][0])
        border = self._params["border_size"]
        if isinstance(border, int):
            pads = [(border, border)] * len(shape)
        elif len(border) == 2 and all(isinstance(x, int) for x in border):
            pads = [tuple(border)] * len(shape)
        else:
            pads = [tuple(pair) for pair in border]
        return {util.DEFAULT_OUTPUT_SLOT: tuple(size + before + after for size, (before, after) in zip(shape, pads))}
