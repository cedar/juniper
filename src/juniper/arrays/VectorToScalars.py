import logging
from ..core.frontend.Step import Step
from ..util import util


logger = logging.getLogger(__name__)
def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        input_vector = input_mats[util.DEFAULT_INPUT_SLOT]
        output = {}
        for i in range(params["N_scalars"]):
            output[f'out{i}'] = input_vector[i]
        return output
    return compute_kernel

class VectorToScalars(Step):
    """
    Description
    ---------
    Turns a 1d-Array (Vector) into a set of individual scalars.

    Parameters
    ---------
    - N_scalars: int 
        - Number of scalars (length of input Vector)

    Step Input/Output slots
    ---------
    - in0: jnp.ndarray(N_scalars)
        - 1d-Array of length N_scalars
    - [out0, out1, ..., out{N_scalars-1}]: jnp.ndarray((1,))
        - separate outputs indexed by 'out{i}'
    """

    def __init__(self, name : str, N_scalars : int):
        params = locals().copy()
        mandatory_params = ["N_scalars"]
        super().__init__(name, params, mandatory_params)

        for i in range(1, self._params["N_scalars"]):
            self.register_output_slot(f'out{i}')

        self.compute_kernel = compute_kernel_factory(self._params)

    def infer_output_shapes(self, input_specs):
        if util.DEFAULT_INPUT_SLOT not in input_specs:
            return {}
        scalar_shape = tuple(input_specs[util.DEFAULT_INPUT_SLOT][0][1:])
        return {f"out{i}": scalar_shape for i in range(self._params["N_scalars"])}
