from .Step import Step
from .. import util
import jax.numpy as jnp
import jax
from functools import partial

class Sum(Step):

    def __init__(self, name, params):
        mandatory_params = []
        super().__init__(name, params, mandatory_params)
        self.needs_input_connections = True
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = jnp.inf

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        # input sum is computed in step.update_input()
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        # Return output
        return {util.DEFAULT_OUTPUT_SLOT: input}

