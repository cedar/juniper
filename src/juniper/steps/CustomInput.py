from .Step import Step
from .. import util
import jax.numpy as jnp

class CustomInput(Step):

    def __init__(self, name, params):
        mandatory_params = ["shape"]
        super().__init__(name, params, mandatory_params)
        
        self.is_source = True

        self.output = jnp.zeroes(params["shape"])
        
    def compute(self, input_mats, **kwargs):
        return {util.DEFAULT_OUTPUT_SLOT: self.output}