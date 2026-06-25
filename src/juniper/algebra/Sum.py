import logging
from ..core.frontend.Step import Step
from ..util import util
import numpy as np


logger = logging.getLogger(__name__)
def compute_kernel_factory():
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        return {util.DEFAULT_OUTPUT_SLOT: input}
    return compute_kernel


class Sum(Step):
    """
    Description
    ---------
    Adds incoming steps component wise.

    Parameters
    ----------

    Step Input/Output slots
    ----------
    - in0 : jnp.ndarray 
    - out0 : jnp.ndarray 
    """
    def __init__(self, name : str):
        params = locals().copy()
        mandatory_params = []
        super().__init__(name, params, mandatory_params)
        self.needs_input_connections = True
        self.set_max_incoming_connections(util.DEFAULT_INPUT_SLOT, np.inf)

        self.compute_kernel = compute_kernel_factory()
