from ..core.frontend.Source import Source
from ..util import util
import jax.numpy as jnp

def compute_kernel_factory():
    return lambda input_mats, buffer, **kwargs: {util.DEFAULT_OUTPUT_SLOT: buffer[util.DEFAULT_OUTPUT_SLOT]}

class CustomInput(Source):
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
    def __init__(self, name : str, shape : tuple):
        params = locals().copy()
        mandatory_params = ["shape"]
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.read_from_cpu = True

        self.output = jnp.zeros(self._params["shape"])
        self.compute_kernel = compute_kernel_factory()
        
    def set_data(self, data):
        self.output = data

    def get_data(self):
        return self.output
