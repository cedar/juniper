import jax
import jax.numpy as jnp
from functools import partial
from src.steps.Step import Step
from src import util


class ExpandAxes(Step):

    def __init__(self, name, params):
        mandatory_params = ["axis", "sizes"]
        super().__init__(name, params, mandatory_params)
        

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = jnp.expand_dims(input, axis=self._params["axis"])

        for ax, size in zip(self._params["axis"], self._params["sizes"]):
            output = jnp.repeat(output, size, axis=ax)
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
    