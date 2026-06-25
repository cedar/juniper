import logging
from ..core.frontend.Step import Step
from ..util import util
import numpy as np
import jax.numpy as jnp


logger = logging.getLogger(__name__)
def compute_kernel_factory(params):
    def compute_kernel(input_mats : dict[str,jnp.ndarray], buffer : dict[str,jnp.ndarray], **kwargs) -> dict[str,jnp.ndarray]:
        # input prod is computed in update_input
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        input = input.astype(params["jdtype"])

        # Return output
        return {util.DEFAULT_OUTPUT_SLOT: input}
    return compute_kernel


class ComponentMultiply(Step):
    """
    Description
    ---------
    Componentwise multiplication of incoming steps.

    Parameters
    ---------

    Step Input/Output slots
    ---------
    - in0 : jnp.array()
    - out0 : jnp.array()
    """
    def __init__(self, name : str):
        params = locals().copy()
        mandatory_params = []
        super().__init__(name, params, mandatory_params)
        self.needs_input_connections = True
        self.set_max_incoming_connections(util.DEFAULT_INPUT_SLOT, np.inf)
        self.input_aggregation = "product"
        self.compute_kernel = compute_kernel_factory(self._params)
