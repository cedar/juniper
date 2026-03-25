


import jax.numpy as jnp
from ..configurables.Step import Step
from ..util import util

def rgb_to_hsv(rgb):
    """Convert RGB image (H,W,3) in [0,1] to HSV in [0,1] using JAX."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    maxc = jnp.max(rgb, axis=-1)
    minc = jnp.min(rgb, axis=-1)
    delta = maxc - minc

    # Hue
    hue = jnp.where(
        delta == 0,
        0.0,
        jnp.where(
            maxc == r,
            (g - b) / (delta + 1e-6),
            jnp.where(
                maxc == g,
                2.0 + (b - r) / (delta + 1e-6),
                4.0 + (r - g) / (delta + 1e-6),
            ),
        ),
    )
    hue = (hue / 6.0) % 1.0  # normalize to [0,1)

    # Saturation
    sat = jnp.where(maxc == 0, 0.0, delta / (maxc + 1e-6))

    # Value
    val = maxc

    return hue, sat, val


def compute_kernel_factory(params):
    sat_threshold = float(params.get("saturation_threshold", 0.2))  # still [0,1]
    val_threshold = float(params.get("value_threshold", 0.2))       # still [0,1]

    def compute_kernel(input_mats, buffer, **kwargs):
        rgb_in = input_mats[util.DEFAULT_INPUT_SLOT]  # (H,W,3) in [0,255] or [0,1]

        # Normalize to [0,1] float for HSV math
        rgb = jnp.asarray(rgb_in, dtype=jnp.float32)
        is_255_range = jnp.max(rgb) > 1.0
        rgb01 = jnp.where(is_255_range, rgb / 255.0, rgb)

        # Convert RGB → HSV in [0,1]
        hue, sat, val = rgb_to_hsv(rgb01)

        # Mask greys/whites/blacks (in HSV space)
        mask = (sat < sat_threshold) | (val < val_threshold)

        # Replace masked pixels with white (in [0,1])
        out01 = jnp.where(mask[..., None], 1.0, rgb01)

        # Convert back to original range/dtype
        out = jnp.where(
            is_255_range,
            jnp.clip(out01 * 255.0, 0.0, 255.0),
            out01
        )
        out = out.astype(rgb_in.dtype)  # keep uint8 if input was uint8

        return {util.DEFAULT_OUTPUT_SLOT: out}

    return compute_kernel

    return compute_kernel


class RemoveBlackWhiteGreys(Step):
    """
    A JAX-accelerated step that removes black, white, and grey pixels 
    (based on value + saturation thresholds) and replaces them with white.
    """

    def __init__(self, name, params):
        mandatory_params = []
        super().__init__(name, params, mandatory_params)

        # Default threshold values if not provided
        if "saturation_threshold" not in self._params:
            self._params["saturation_threshold"] = 0.2
        if "value_threshold" not in self._params:
            self._params["value_threshold"] = 0.2

        self.compute_kernel = compute_kernel_factory(self._params)
