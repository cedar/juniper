


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
    sat_threshold = float(params.get("saturation_threshold", 0.2))
    val_threshold = float(params.get("value_threshold", 0.2))

    def compute_kernel(input_mats, buffer, **kwargs):
        rgb = input_mats[util.DEFAULT_INPUT_SLOT]  # (H,W,3) float32 in [0,1]

        # Convert RGB → HSV
        hue, sat, val = rgb_to_hsv(rgb)

        # Identify blacks, whites, greys:
        # - Low saturation → grey
        # - Low value → black
        # - High value + low saturation → white-ish but treated as grey/white
        mask = (sat < sat_threshold) | (val < val_threshold)

        # Replace those pixels with white
        # mask shape: (H,W) → broadcast to (H,W,3)
        mask3 = jnp.repeat(mask[..., None], 3, axis=-1)
        white = jnp.ones_like(rgb)

        out = jnp.where(mask3, white, rgb)

        return {util.DEFAULT_OUTPUT_SLOT: out}

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
