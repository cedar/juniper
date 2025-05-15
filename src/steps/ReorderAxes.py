import jax
import jax.numpy as jnp
from functools import partial
from src.steps.Step import Step
from src import util

class ReorderAxes(Step):

    def __init__(self, name, params):
        mandatory_params = ["order"]
        super().__init__(name, params, mandatory_params)
        

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = jnp.transpose(input, axes=self._params["order"])
        return {util.DEFAULT_OUTPUT_SLOT: output}
    