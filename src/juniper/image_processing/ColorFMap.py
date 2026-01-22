from ..configurables.Step import Step
from ..util import util

import jax.numpy as jnp
import jax.nn as jnn

def hsv_to_color_indices(hue, sat, val, sat_threshold=0.2, value_threshold=0.2):
    """
    hue, sat, val: arrays shaped (H, W), hue in [0, 1)
    returns: int32 labels shaped (H, W)
             0 = masked/background
             1..6 = {red, orange, yellow, green, blue, purple}
    """
    # Hue bucketization (1..6)
    red = (hue < 0.04) | (hue >= 0.90)
    orange = (hue >= 0.04) & (hue < 0.10)
    yellow = (hue >= 0.10) & (hue < 0.20)
    green = (hue >= 0.20) & (hue < 0.45)
    blue = (hue >= 0.45) & (hue < 0.70)
    purple = (hue >= 0.70) & (hue < 0.90)

    # start with "purple" as default, then override with earlier buckets
    labels = jnp.full(hue.shape, 6, dtype=jnp.int32)
    labels = jnp.where(blue,   5, labels)
    labels = jnp.where(green,  4, labels)
    labels = jnp.where(yellow, 3, labels)
    labels = jnp.where(orange, 2, labels)
    labels = jnp.where(red,    1, labels)

    # Mask low-sat / low-value -> background (0)
    valid = (sat >= sat_threshold) & (val >= value_threshold)
    labels = jnp.where(valid, labels, 0)

    return labels


def hsv_to_onehot(hue, sat, val, sat_threshold=0.2, value_threshold=0.2, num_colors=6):
    """
    Returns onehot (H, W, num_colors) for the 6 hue buckets.
    Background is removed (masked pixels -> all zeros).
    """
    labels = hsv_to_color_indices(hue, sat, val, sat_threshold, value_threshold)
    # one_hot needs non-negative ints; we used 0..6 with 0=background
    oh = jnn.one_hot(labels, num_classes=num_colors + 1)  # (H,W,7)
    return oh[..., 1:]  # drop background channel -> (H,W,6)


def compute_kernel_factory(params):
    sat_threshold = float(params.get("saturation_threshold", 0.2))
    val_threshold = float(params.get("value_threshold", 0.2))

    def compute_kernel(input_mats, buffer, **kwargs):
        # Expect these inputs to already be HSV channels (0..1 range).
        # hue in DEFAULT_INPUT_SLOT, sat in "in1", val in "in2" (adjust names if yours differ)
        hue = input_mats[util.DEFAULT_INPUT_SLOT]
        sat = input_mats["in1"]
        val = input_mats["in2"]

        out = hsv_to_onehot(
            hue, sat, val,
            sat_threshold=sat_threshold,
            value_threshold=val_threshold,
            num_colors=6,
        )  # (H,W,6)

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

