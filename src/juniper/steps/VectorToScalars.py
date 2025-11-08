import jax
from functools import partial
from .Step import Step
from .. import util


class VectorToScalars(Step):
    """
    Turns a 1d-Array (Vector) into a set of individual scalars.

    Parameters
    ---------
    - N_scalars: int 
        - Number of scalars (length of input Vector)

    Step Computation
    ---------
    - Input: jnp.ndarray 
        - 1d-Array of length N_scalars
    - output: jnp.ndarray 
        - separate outputs indexed by 'out{i}'
    """

    def __init__(self, name, params):
        mandatory_params = ["N_scalars"]
        super().__init__(name, params, mandatory_params)

        for i in range(1, self._params["N_scalars"]):
            self.register_output(f'out{i}')

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input_vector = input_mats[util.DEFAULT_INPUT_SLOT]
        output = {}
        for i in range(self._params["N_scalars"]):
            output[f'out{i}'] = input_vector[i]
        return output