from ..configurables.Step import Step
from ..util import util

import jax.numpy as jnp
import jax.nn as jnn

def hsv_to_onehot10(hue, sat, val, sat_threshold=0.3, value_threshold=0.3):
    """
    hue, sat, val: (H, W), hue in [0,1)
    returns: (H, W, 10) float32
    channels 0..5 : {red, orange, yellow, green, blue, purple}
    channel 9     : brown
    channels 6..8 : zeros

    invalid pixels (low sat/val): all zeros across all 10 channels
    """

    # --- Basic color regions ---
    red    = (hue < 0.04) | (hue >= 0.99)
    orange = (hue >= 0.04) & (hue < 0.10)
    yellow = (hue >= 0.10) & (hue < 0.15)
    green  = (hue >= 0.15) & (hue < 0.20)
    blue   = (hue >= 0.20) & (hue < 0.70)
    purple = (hue >= 0.70) & (hue < 0.99)

    # --- Brown: low-value orange–yellow tones ---
    # Hue: 0.05–0.15, Sat >= 0.3, Value <= 0.5
    brown = ((hue >= 0.05) & (hue < 0.15)
            & (sat >= 0.3)
            & (val <= 0.5))

    # default label = purple (index 5)
    labels = jnp.full(hue.shape, 5, dtype=jnp.int32)
    labels = jnp.where(blue,   4, labels)
    labels = jnp.where(green,  3, labels)
    labels = jnp.where(yellow, 2, labels)
    labels = jnp.where(orange, 1, labels)
    labels = jnp.where(red,    0, labels)

    # Brown overrides orange/yellow
    labels = jnp.where(brown, 6, labels)  # temporarily assign brown = 6

    # mask invalid pixels
    valid = (sat >= sat_threshold) & (val >= value_threshold)
    labels = jnp.where(valid, labels, -1)

    # Convert to one-hot: we now need 7 classes (0–6)
    shifted = labels + 1               # -1→0, 0→1, ..., 6→7
    oh7 = jnn.one_hot(shifted, num_classes=8, dtype=jnp.float32)

    # slice out classes 1–7 (ignore background class 0)
    oh = oh7[..., 1:]                 # shape (H,W,7)

    # oh channels now = [red, orange, yellow, green, blue, purple, brown]

    # Now remap into a 10-channel output:
    # 0–5  = original 6 colors
    # 6–8  = zeros
    # 9    = brown
    base6 = oh[..., :6]                # (H,W,6)
    brown_ch = oh[..., 6:7]            # (H,W,1)

    zeros3 = jnp.zeros_like(base6[..., :3])  # (H,W,3)

    # final = 6 + 3 + 1 = 10 channels
    out10 = jnp.concatenate([base6, zeros3, brown_ch], axis=-1)
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

