import jax
import jax.numpy as jnp
from functools import partial
from .Step import Step
from .. import util

COMPRESSION_TYPE_MAP = {
    "Sum": jnp.sum,
    "Average": jnp.average,
    "Maximum": jnp.max,
    "Minimum": jnp.min,
}

class CompressAxes(Step):

    def __init__(self, name, params):
        mandatory_params = ["axis", "compression_type"]
        super().__init__(name, params, mandatory_params)
        try:
            self._red_func = COMPRESSION_TYPE_MAP[self._params["compression_type"]]
        except KeyError:
            raise ValueError(
                f"Unknown compression type: {self._params['compression_type']}. "
                f"Supported compression types are: {', '.join(COMPRESSION_TYPE_MAP)}"
                )

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = self._red_func(input, axis=self._params["axis"])
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
    