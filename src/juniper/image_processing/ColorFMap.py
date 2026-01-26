from ..configurables.Step import Step
from ..util import util

import jax.numpy as jnp
import jax.nn as jnn

def hsv_to_onehot10(hue, sat, val, sat_threshold=0.3, value_threshold=0.3):
    """
    hue, sat, val: (H, W), hue in [0,1)
    returns: (H, W, 10) float32
      channels 0..5 : {red, orange, yellow, green, blue, purple}
      channels 6..9 : zeros
      invalid pixels (low sat/val): all zeros across all 10 channels
    """
    # boolean buckets (same boundaries as your numpy code)
    red    = (hue < 0.04) | (hue >= 0.90)
    orange = (hue >= 0.04) & (hue < 0.10)
    yellow = (hue >= 0.10) & (hue < 0.15)
    green  = (hue >= 0.15) & (hue < 0.20)
    blue   = (hue >= 0.20) & (hue < 0.70)
    purple = (hue >= 0.70) & (hue < 0.90)

    # Build labels in 0..5
    labels = jnp.full(hue.shape, 5, dtype=jnp.int32)  # default purple=5
    labels = jnp.where(blue,   4, labels)
    labels = jnp.where(green,  3, labels)
    labels = jnp.where(yellow, 2, labels)
    labels = jnp.where(orange, 1, labels)
    labels = jnp.where(red,    0, labels)

    # Mask invalid pixels to -1 (like your numpy code)
    valid = (sat >= sat_threshold) & (val >= value_threshold)
    labels = jnp.where(valid, labels, -1)

    # Convert to onehot with "background" handling:
    # shift by +1 so invalid (-1) -> 0
    shifted = labels + 1  # now in {0..6}
    oh7 = jnn.one_hot(shifted, num_classes=7, dtype=jnp.float32)  # (H,W,7)
    oh6 = oh7[..., 1:]  # drop background -> (H,W,6), invalid pixels are all zeros

    # Pad to 10 channels (last 4 are zeros)
    out10 = jnp.pad(oh6, pad_width=((0, 0), (0, 0), (0, 4)), mode="constant")
    return out10


def compute_kernel_factory(params):
    sat_threshold = float(params.get("saturation_threshold", 0.2))
    val_threshold = float(params.get("value_threshold", 0.2))

    def compute_kernel(input_mats, buffer, **kwargs):
        hue = input_mats[util.DEFAULT_INPUT_SLOT]  # (H,W), 0..1
        sat = input_mats["in1"]                   # (H,W), 0..1
        val = input_mats["in2"]                   # (H,W), 0..1  <-- adjust key if needed

        out = hsv_to_onehot10(
            hue, sat, val,
            sat_threshold=sat_threshold,
            value_threshold=val_threshold,
        )  # (H,W,10)

        return {util.DEFAULT_OUTPUT_SLOT: out}

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

        if "value_threshold" not in self._params.keys():
            self._params["value_threshold"] = 0

        self.register_input("in1")
        self.register_input("in2")

