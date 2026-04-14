import jax
import jax.numpy as jnp
from functools import partial
from ..configurables.Step import Step
from ..util import util

COMPRESSION_TYPE_MAP = {
    "Sum": jnp.sum,
    "Average": jnp.average,
    "Maximum": jnp.max,
    "Minimum": jnp.min,
}

def compute_kernel_factory(params, red_func):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = red_func(input, axis=params["axis"])
        if params["compress_all"]:
            output = jnp.array([output])
        return {util.DEFAULT_OUTPUT_SLOT: output}

    return compute_kernel


class CompressAxes(Step):
    """
    Description
    ---------
    Compress incoming step along specified dimension.

    Parameters
    ---------
    - axis : tuple(ax0,ax1,...)
    - compression_type : str(Sum,Average,Maximum,Minimum)
    - compress_all (optional) : bool
        - flag to indicate that all input axes will be supressed. This is needed to establish valid output shape.

    Step Input/Output slots
    ---------
    - in0 : jnp.array()
    - out0 : jnp.array()
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["axis", "compression_type"]
        super().__init__(name, params, mandatory_params)
        try:
            self._red_func = COMPRESSION_TYPE_MAP[self._params["compression_type"]]
        except KeyError:
            raise ValueError(
                f"Unknown compression type: {self._params['compression_type']}. "
                f"Supported compression types are: {', '.join(COMPRESSION_TYPE_MAP)}"
                )
        
        if "compress_all" not in params.keys():
            self._params["compress_all"] = False

        self.compute_kernel = compute_kernel_factory(self._params, self._red_func)

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, buffer, **kwargs):
        return self.compute_kernel(input_mats, buffer, **kwargs)
    