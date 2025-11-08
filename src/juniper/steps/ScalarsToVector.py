import jax
import jax.numpy as jnp
from functools import partial
from .Step import Step
from .. import util


class ScalarsToVector(Step):
    """
    Turns a number of scalars into a 1d-Array (vector).

    TODO: Make it possible to have incomplete incoming connections.

    Parameters
    ----------
    - N_scalars: int 
        - Number of scalars (length of output Vector)

    Step Computation
    ----------
    - Input: jnp.ndarray 
        - N_scalars separate inputs indexed by 'in{i}'
    - output: jnp.ndarray 
        - Vector of length N_scalars
    """

    def __init__(self, name, params):
        mandatory_params = ["N_scalars"]
        super().__init__(name, params, mandatory_params)

        for i in range(1, self._params["N_scalars"]):
            self.register_input(f'in{i}')

        self.needs_input_connections == False

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        output = jnp.zeros((self._params["N_scalars"]))
        for i in range(self._params["N_scalars"]):
            output = output.at[i].set(input_mats[f'in{i}'][0])
        return {util.DEFAULT_OUTPUT_SLOT: output}