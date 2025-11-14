from ..configurables.Step import Step
from ..util import util
import jax.numpy as jnp

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

        self.output = jnp.zeros(self._params["shape"])
        
    def compute(self, input_mats, **kwargs):
        return {util.DEFAULT_OUTPUT_SLOT: self.output}