import jax
import jax.numpy as jnp
from functools import partial
from src.steps.Step import Step
from src import util

@partial(jax.jit, static_argnames=["axis"])
def Sum(x, axis):
    return jnp.sum(x, axis=axis)
@partial(jax.jit, static_argnames=["axis"])
def Average(x, axis):
    return jnp.average(x, axis=axis)
@partial(jax.jit, static_argnames=["axis"])
def Maximum(x, axis):
    return jnp.max(x, axis=axis)
@partial(jax.jit, static_argnames=["axis"])
def Minimum(x, axis):
    return jnp.min(x, axis=axis)

def Sum_wrapper(axis):
    return lambda x: Sum(x, axis)
def Average_wrapper(axis):
    return lambda x: Average(x, axis)
def Maximum_wrapper(axis):
    return lambda x: Maximum(x, axis)
def Minimum_wrapper(axis):
    return lambda x: Minimum(x, axis)

class CompressAxes(Step):

    def __init__(self, name, params):
        mandatory_params = ["axis", "compression_type"]
        super().__init__(name, params, mandatory_params)
        if self._params["compression_type"] == "Sum":
            self._red_func = Sum_wrapper(self._params["axis"])
        elif self._params["compression_type"] == "Average":
            self._red_func = Average_wrapper(self._params["axis"])
        elif self._params["compression_type"] == "Maximum":
            self._red_func = Maximum_wrapper(self._params["axis"])
        elif self._params["compression_type"] == "Minimum":
            self._red_func = Minimum_wrapper(self._params["axis"])
        else:
            raise ValueError(f"Unknown compression type: {self._params['compression_type']}. Supported functions are: "
                             "Sum, Average, Maximum, Minimum.")

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = self._red_func(input)
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
    