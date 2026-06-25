import logging
from ..core.frontend.Step import Step
from ..util import util

import jax.numpy as jnp
import jax.nn as jnn


logger = logging.getLogger(__name__)
def hsv_to_onehot10(hue, sat, val, sat_threshold=0.3, value_threshold=0.3):
    """
    hue, sat, val: (H, W), hue in [0,1)
    returns: (H, W, 10) float32
      channels 0..5 : {red, orange, yellow, green, blue, purple}
      channels 6..9 : zeros
      invalid pixels (low sat/val): all zeros across all 10 channels
    """
    # boolean buckets (same boundaries as your numpy code)
    red    = (hue < 0.04) | (hue >= 0.96)
    orange = (hue >= 0.04) & (hue < 0.10)
    yellow = (hue >= 0.10) & (hue < 0.18)
    green  = (hue >= 0.18) & (hue < 0.25)
    blue   = (hue >= 0.25) & (hue < 0.70)
    purple = (hue >= 0.70) & (hue < 0.96)  # noqa: F841

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
    """
    Description
    ---------
    Converts hue, saturation and value maps into a 10-channel color feature map.
    The first six channels encode red, orange, yellow, green, blue and purple as
    one-hot activations. Pixels below the saturation or value threshold are
    suppressed.

    Parameters
    ---------
    - bins : int
        - Number of color bins expected by the feature map interface.
    - saturation_threshold (optional) : float
        - Minimum saturation required for a pixel to activate a color channel.
        - Default = 0.2
    - hue_range (optional) : int
        - Hue range metadata.
        - Default = 360
    - value_threshold (optional) : float
        - Minimum value required for a pixel to activate a color channel.
        - Default = 0.2

    Step Input/Output slots
    ---------
    - in0: jnp.array((H,W))
        - Hue channel in [0,1].
    - in1: jnp.array((H,W))
        - Saturation channel in [0,1].
    - in2: jnp.array((H,W))
        - Value channel in [0,1].
    - out0: jnp.array((H,W,10))
    """

    _saturation_threshold = 0.2
    _hue_range = 360
    _value_threshold = 0.2
    def __init__(
            self,
            name : str,
            bins : int,
            saturation_threshold : float = _saturation_threshold,
            hue_range : int = _hue_range,
            value_threshold : float = _value_threshold):
        params = locals().copy()
        mandatory_params = ["bins"] 
        super().__init__(name, params, mandatory_params)
        self.compute_kernel = compute_kernel_factory(self._params)

        self.register_input_slot("in1")
        self.register_input_slot("in2")

    def infer_output_shapes(self, input_specs):
        if util.DEFAULT_INPUT_SLOT not in input_specs:
            return {}
        hue_shape = input_specs[util.DEFAULT_INPUT_SLOT][0]
        return {util.DEFAULT_OUTPUT_SLOT: tuple(hue_shape) + (10,)}
