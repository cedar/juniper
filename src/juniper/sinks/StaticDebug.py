from ..core.Step import Step
from ..util import util
import jax.numpy as jnp


def compute_kernel_factory():
    def compute_kernel(input_mats, buffer, **kwargs):
        return {}
    return compute_kernel

class StaticDebug(Step):
    """
    Description
    ---------
    An empty dynamic step to be used as a sink for static steps to force their computation even when not connected to any field.

    Parameters

    Step Input/Output slots
    ---------
    - Input: any
    - output: any
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["shape"]
        params["shape"] = (1,)
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.needs_input_connections = True
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = jnp.inf
        
        self.compute_kernel = compute_kernel_factory()

    def update_input(self, arch, input_slot_shape="shape"):
        return {util.DEFAULT_INPUT_SLOT: jnp.zeros(self._params["shape"])}
    
    def set_data(self, data):
        pass
    
