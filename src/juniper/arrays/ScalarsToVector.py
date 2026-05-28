import jax.numpy as jnp
from ..configurables.Step import Step
from ..util import util

def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        output = jnp.zeros((params["N_scalars"],))
        for i in range(params["N_scalars"]):
            if input_mats[f'in{i}'] is not None:
                output = output.at[i].set(jnp.squeeze(input_mats[f'in{i}']))
        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel

class ScalarsToVector(Step):
    """
    Description
    ---------
    Turns a number of scalars into a 1d-Array (vector).

    TODO: Make it possible to have incomplete incoming connections.

    Parameters
    ----------
    - N_scalars: int 
        - Number of scalars (length of output Vector)

    Step Input/Output slots
    ----------
    - [in0, in1, ..., in{N_scalars-1}] : jnp.ndarray 
        - N_scalars separate inputs indexed by 'in{i}'
    - out0 : jnp.ndarray 
        - Vector of length N_scalars
    """

    def __init__(self, name : str, params : dict):
        mandatory_params = ["N_scalars"]
        super().__init__(name, params, mandatory_params)

        for i in range(1, self._params["N_scalars"]):
            self.register_input_slot(f'in{i}')

        self.needs_input_connections = False
        self.compute_kernel = compute_kernel_factory(self._params)

    def infer_output_shapes(self, input_specs):
        return {util.DEFAULT_OUTPUT_SLOT: (self._params["N_scalars"],)}
