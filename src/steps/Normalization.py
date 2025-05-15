import jax
import jax.numpy as jnp
from functools import partial
from src.steps.Step import Step
from src import util

@jax.jit
def L2_norm(x):
    return jnp.linalg.norm(x, ord=2)
@jax.jit
def L1_norm(x):
    return jnp.linalg.norm(x, ord=1)
@jax.jit
def Infinity_norm(x):
    return jnp.linalg.norm(x, ord=jnp.inf)

def L2_norm_wrapper():
    return lambda x: L2_norm(x)
def L1_norm_wrapper():
    return lambda x: L1_norm(x)
def Infinity_norm_wrapper():
    return lambda x: Infinity_norm(x)

class Normalization(Step):

    def __init__(self, name, params):
        mandatory_params = ["function"]
        super().__init__(name, params, mandatory_params)
        if self._params["function"] == "InfinityNorm":
            self._norm_func = Infinity_norm_wrapper()
        elif self._params["function"] == "L1Norm":
            self._norm_func = L1_norm_wrapper()
        elif self._params["function"] == "L2Norm":
            self._norm_func = L2_norm_wrapper()
        else:
            raise ValueError(f"Unknown function: {self._params['function']}. Supported functions are: "
                             "InfinityNorm, L1Norm, L2Norm.")

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        epsilon = 1e-8 # to avoid division by zero
        output = input / (epsilon + self._norm_func(input))
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
    