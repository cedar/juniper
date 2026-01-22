from ..configurables.Step import Step
from ..util import util

import jax.numpy as jnp
import jax.nn as jnn

def compute_kernel_factory(params):
    saturation_threshold = params["saturation_threshold"]
    hue_range = params["hue_range"]
    def compute_kernel(input_mats, buffer, **kwargs):
        hue = input_mats[util.DEFAULT_INPUT_SLOT] 
        sat = input_mats["in1"]
        mask = sat > saturation_threshold
        hue_deg = hue * 360.0
        num_bins = int(params["bins"]) + 2 # will change this next
        bin_size = hue_range / num_bins
        bin_indices = jnp.floor(hue_deg / bin_size)
        bin_indices = jnp.clip(bin_indices, 0, num_bins - 1)
        masked_bin_indices = jnp.where(mask, bin_indices, 0)
        out = jnn.one_hot(masked_bin_indices, num_bins)
        return {util.DEFAULT_OUTPUT_SLOT: out[:,:,1:num_bins-1]}
    return compute_kernel


class ColorFMap(Step):

    def __init__(self, name, params):
        mandatory_params = ["bins"] 
        super().__init__(name, params, mandatory_params)
        self.compute_kernel = compute_kernel_factory(self._params)
        
        if "saturation_threshold" not in self._params.keys():
            self._params["saturation_threshold"] = 0

        if "hue_range" not in self._params.keys():
            self._params["hue_range"] = 360

        self.register_input("in1")

