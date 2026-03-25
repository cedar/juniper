from ...configurables.Step import Step
from functools import partial
from ...util import util
import jax.numpy as jnp
import jax


def compute_kernel_factory(params):
    def az_tilt_dirs(az, tilt):
        ca, sa = jnp.cos(az), jnp.sin(az)
        ct, st = jnp.cos(tilt), jnp.sin(tilt)
        x = st * ca
        y = st * sa
        z = ct
        return jnp.stack([x, y, z], axis=-1)

    def compute_kernel(input_mats, buffer, **kwargs):
        range_img = jnp.asarray(input_mats[util.DEFAULT_INPUT_SLOT], dtype=jnp.float32)

        n_tilt, n_pan = params["image_shape"]
        pan_lo, pan_hi = params["pan_range"]
        tilt_lo, tilt_hi = params["tilt_range"]

        pan_centers  = jnp.linspace(pan_lo,  pan_hi,  n_pan,  endpoint=False) + 0.5 * (pan_hi  - pan_lo ) / n_pan
        tilt_centers = jnp.linspace(tilt_lo, tilt_hi, n_tilt, endpoint=False) + 0.5 * (tilt_hi - tilt_lo) / n_tilt

        az   = jnp.broadcast_to(pan_centers,           (n_tilt, n_pan))
        tilt = jnp.broadcast_to(tilt_centers[:, None], (n_tilt, n_pan))

        dirs = az_tilt_dirs(az, tilt)                  # (n_tilt, n_pan, 3)
        vecs = dirs * range_img[..., None]             # (n_tilt, n_pan, 3)

        return {util.DEFAULT_OUTPUT_SLOT: vecs.reshape(-1, 3)}

    return compute_kernel


class RangeImageToVectors(Step):
    """
    Description
    ---------
    Converts a range image (in spherical coordinates) into a set of vectors.

    Parameters
    -----------
      - pan_range (azimuth) : [pan_low, pan_high]
      - tilt_range (polar) :  [tilt_low, tilt_high]
      - image_shape : (n_tilt, n_pan)  [Y, X]

    Step Input/Output slots
    ---------
    - in0: jnp.ndarray(image_shape)
    - out0: jnp.ndarray(H*W,3)
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["image_shape", "pan_range", "tilt_range"]
        super().__init__(name, params, mandatory_params)
        self.compute_kernel = compute_kernel_factory(self._params)

    
