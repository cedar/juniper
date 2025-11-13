from .Step import Step
from functools import partial
from .. import util
import jax.numpy as jnp
import jax


class VectorsToRangeImage(Step):
    """
    Description
    ---------
    Converts a set of Vectors into a range image.

    Parameters
    -----------
      - pan (azimuth) : [pan_low, pan_high]
      - tilt (polar) :  [tilt_low, tilt_high]
      - image_shape : (n_tilt, n_pan)  [Y, X]

    Step Input/Output slots
    ---------
    - in0: jnp.ndarray(H*W,3)
    - out0: jnp.ndarray(image_shape)
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["image_shape", "pan_range", "tilt_range"]
        super().__init__(name, params, mandatory_params)
    
    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):

        v = jnp.asarray(input_mats[util.DEFAULT_INPUT_SLOT], dtype=jnp.float32)

        n_pan, n_tilt = self._params["image_shape"]
        pan_lo, pan_hi = self._params["pan_range"]
        tilt_lo, tilt_hi = self._params["tilt_range"]

        # transform to spherical
        eps = 1e-8
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        rho = jnp.sqrt(x*x + y*y) + eps
        r = jnp.sqrt(x * x + y * y + z * z) + eps

        pan = jnp.arctan2(y, x)
        tilt = jnp.arcsin(jnp.clip(z / r, -1.0, 1.0))#jnp.arctan2(z, rho)#

        # generate bins in range image
        scale = n_pan / (pan_hi - pan_lo)
        ix = jnp.floor((pan - pan_lo) * scale).astype(jnp.int32)
        ix = jnp.clip(ix, 0, n_pan - 1)                 # X

        scale = n_tilt / (tilt_hi - tilt_lo)
        iy = jnp.floor((tilt - tilt_lo) * scale).astype(jnp.int32)
        iy = jnp.clip(iy, 0, n_tilt - 1)                   # Y

        # generate image and fill out blank spots
        img = jnp.full((n_tilt, n_pan), jnp.inf, dtype=jnp.float32)
        img = img.at[iy, ix].min(r)
        img = jnp.where(jnp.isinf(img), 0, img) # fill up missing pixels with 0

        return {util.DEFAULT_OUTPUT_SLOT: img}

    
