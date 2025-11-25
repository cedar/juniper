import jax
from functools import partial
from ..configurables.Step import Step
from ..util import util

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

    def __init__(self, name : str, params : dict):
        mandatory_params = ["N_scalars"]
        super().__init__(name, params, mandatory_params)

        for i in range(1, self._params["N_scalars"]):
            self.register_output(f'out{i}')

        self.compute_kernel = compute_kernel_factory(self._params)

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, buffer, **kwargs):
        return self.compute_kernel(input_mats, buffer, **kwargs)