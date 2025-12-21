from ..configurables.Step import Step
from ..util import util

import jax.numpy as jnp
import jax.nn as jnn

def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        hue = input_mats[util.DEFAULT_INPUT_SLOT] 
        hue_deg = hue * 360.0
        num_bins = int(params["bins"])
        bin_size = 360.0 / num_bins
        bin_indices = jnp.floor(hue_deg / bin_size)
        bin_indices = jnp.clip(bin_indices, 0, num_bins - 1)
        return {util.DEFAULT_OUTPUT_SLOT: jnn.one_hot(bin_indices, num_bins)}
    return compute_kernel


class ColorFMap(Step):

    def __init__(self, name, params):
        mandatory_params = ["bins"] 
        super().__init__(name, params, mandatory_params)
        self.compute_kernel = compute_kernel_factory(self._params)

