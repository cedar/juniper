from ..configurables.Step import Step
from ..util import util
import jax.numpy as jnp
import jax.debug as jgdb

# JAX rgb -> hsv, expects rgb in [0, 1], shape (..., 3)
def rgb_to_hsv_jax(rgb: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    cmax = jnp.maximum(jnp.maximum(r, g), b)
    cmin = jnp.minimum(jnp.minimum(r, g), b)
    delta = cmax - cmin

    # Value
    v = cmax

    # Saturation
    s = jnp.where(cmax > eps, delta / (cmax + eps), 0.0)

    # Hue (in [0,1))
    # Compute raw hue in "sextants" then normalize by /6.
    # Handle delta==0 separately -> hue = 0
    rc = (cmax - r) / (delta + eps)
    gc = (cmax - g) / (delta + eps)
    bc = (cmax - b) / (delta + eps)

    h_r = (bc - gc)              # when cmax == r
    h_g = 2.0 + (rc - bc)        # when cmax == g
    h_b = 4.0 + (gc - rc)        # when cmax == b

    # Select based on which channel is max
    h = jnp.where(cmax == r, h_r, jnp.where(cmax == g, h_g, h_b))
    h = jnp.where(delta > eps, h / 6.0, 0.0)
    h = jnp.mod(h, 1.0)  # wrap to [0,1)

    return jnp.stack([h, s, v], axis=-1)

def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        # Convert to NumPy
        rgb = input_mats[util.DEFAULT_INPUT_SLOT] / 255.0  # shape (H, W, 3)
    
        # Convert to HSV
        hsv = rgb_to_hsv_jax(rgb)  # shape (H, W, 3), values in [0,1]

        return {util.DEFAULT_OUTPUT_SLOT: hsv[:,:,0],
               "out1": hsv[:,:,1],
               "out2": hsv[:,:,2]}
    return compute_kernel


class ColorConversion(Step):
    """
    Description
    ---------
    Converts an RGB image into a HSV image

    Parameters
    ---------

    Step Input/Output slots
    ---------
    - in0 : jnp.array((H,W,3))
    - out0 : jnp.array((H,W))
    - out1 : jnp.array((H,W))
    - out2 : jnp.array((H,W))
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = []
        super().__init__(name, params, mandatory_params)
        
        self.register_output("out1")
        self.register_output("out2")
        self.compute_kernel = compute_kernel_factory(self._params)