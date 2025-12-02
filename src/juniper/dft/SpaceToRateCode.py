from ..configurables.Step import Step
from functools import partial
from ..util import util
import jax.numpy as jnp
import jax

def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):

        input_field = jnp.asarray(input_mats[util.DEFAULT_INPUT_SLOT], dtype=jnp.float32)

        peak_position = jnp.argwhere(input_field > params["threshold"], size = input_field.size)
        N_above_threshold = jnp.sum(jnp.sum(peak_position, axis=1) > 0, axis=0)
        peak_found = jnp.sum(peak_position)>0

        output = jnp.sum(peak_position, axis=0)/N_above_threshold * peak_found + buffer["peak_pos"] * (1-peak_found)

        return {util.DEFAULT_OUTPUT_SLOT: output[0]}
    return compute_kernel
        

class SpaceToRateCode(Step):
    """
    Description
    ---------
    Takes field like array and produces a vector centered at field peak position coordinates. Assumes at most one peak in the input field.

    TODO: make multiple peaks possible? Make cyclic possible? Implement expoential convergance to attractor
    
    Parameters
    ----------
    - shape : tuple(Nx,Ny,...)
    - limits : tuple((lx,ux), (ly,uy), ...)
    - tau (optional) : float
        - If set, the output vector will exponentially converge to the peak position. If not, the vector jumps to the attractor in one time-step
        - Default = 0
    - cyclic (optional) : bool
        - Default = False
    - threshold (optional) : float
        - Default = 0.9

    Step Input/Output slots
    -----------
    - in0 : jnp.array((Nx,Ny,...))
    - out0 : jnp.array(len(Nx,Ny,...))
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ['limits', 'shape']
        super().__init__(name, params, mandatory_params, is_dynamic=True)

        if "tau" not in self._params.keys():
            self._params["tau"] = 0

        if "cyclic" not in self._params.keys():
            self._params["cyclic"] = 0

        if "threshold" not in self._params.keys():
            self._params["threshold"] = 0.9

        self._params["space_dim"] = len(self._params["shape"])

        self._limits = jnp.asarray(self._params["limits"], dtype=jnp.float32)

        self.register_buffer("peak_pos", slot_shape="space_dim")

        self.compute_kernel = compute_kernel_factory(self._params)

        self.reset()


    def reset(self):
        self.buffer[util.DEFAULT_INPUT_SLOT] = jnp.zeros(self._params["shape"])
        self.buffer[util.DEFAULT_OUTPUT_SLOT] = jnp.zeros((len(self._params["shape"]),))
        self.buffer["peak_pos"] = jnp.zeros((len(self._params["shape"]),))


    
