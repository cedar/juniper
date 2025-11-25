from ..configurables.Step import Step
from ..util import util
import jax.numpy as jnp

def compute_kernel_factory():
    return lambda input_mats, buffer, **kwargs: {util.DEFAULT_OUTPUT_SLOT: buffer["output"], "output": buffer["output"]}

class CustomInput(Step):
    """
    Description
    ---------
    Custom Input, can be set from outside by modifying self.output.

    Parameters
    ---------
    - shape : tuple((Nx,Ny,...))

    Step Input/Output slots
    ---------
    - out0 : jnp.array(shape)
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["shape"]
        super().__init__(name, params, mandatory_params)
        
        self.is_source = True
        self.is_exposed = True

        self.output = jnp.zeros(self._params["shape"])
        self.register_buffer("output")
        self.buffer["output"] = self.output
        self.compute_kernel = compute_kernel_factory()
        
    def compute(self, input_mats, buffer, **kwargs):
        return self.compute_kernel(input_mats, buffer, **kwargs)
    
    def reset(self):
        reset_state = {}
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self.buffer["output"]
        reset_state["output"] = self.output
        return reset_state