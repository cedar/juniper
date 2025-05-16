from src.steps.Step import Step
from src import util
import jax
from functools import partial

class ComponentMultiply(Step):

    def __init__(self, name, params):
        mandatory_params = []
        super().__init__(name, params, mandatory_params)
        self.needs_input_connections = True
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = 2

        self.register_input("in1")

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        # input sum is computed in step.update_input()
        output = input_mats[util.DEFAULT_INPUT_SLOT] * input_mats["in1"]

        # Return output
        return {util.DEFAULT_OUTPUT_SLOT: output}

