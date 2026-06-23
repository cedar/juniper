import logging
from ..core.frontend.Step import Step
from ..util import util
import numpy as np



logger = logging.getLogger(__name__)
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
    _shape = (1,)
    def __init__(self, name : str, shape : tuple = _shape):
        params = locals().copy()
        mandatory_params = ["shape"]
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.needs_input_connections = True
        self.set_max_incoming_connections(util.DEFAULT_INPUT_SLOT, np.inf)
        
        self.compute_kernel = compute_kernel_factory()
    
    def set_data(self, data):
        pass
    
