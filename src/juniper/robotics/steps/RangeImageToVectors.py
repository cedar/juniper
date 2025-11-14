from ...configurables.Step import Step
from functools import partial
from ...util import util
import jax.numpy as jnp
import jax


def az_el_dirs(az, el):
    """(azimuth, elevation) -> unit direction (x,y,z) in +x fwd, +y left, +z up."""
    ca, sa = jnp.cos(az), jnp.sin(az)
    ce, se = jnp.cos(el), jnp.sin(el)
    x = ce * ca
    y = ce * sa
    z = se
    return jnp.stack([x, y, z], axis=-1)

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
    
    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        range_img = input_mats[util.DEFAULT_INPUT_SLOT]

        n_tilt, n_pan = self._params["image_shape"]
        pan_lo, pan_hi = self._params["pan_range"]
        tilt_lo, tilt_hi = self._params["tilt_range"]

        # bin centers (uniform)
        pan_centers  = jnp.linspace(pan_lo,  pan_hi,  n_pan,  endpoint=False) + 0.5 * (pan_hi  - pan_lo ) / n_pan
        tilt_centers = jnp.linspace(tilt_lo, tilt_hi, n_tilt, endpoint=False) + 0.5 * (tilt_hi - tilt_lo) / n_tilt

        # per-pixel angles
        az   = jnp.broadcast_to(pan_centers,              (n_tilt, n_pan))
        elev = jnp.broadcast_to(tilt_centers[:, None],    (n_tilt, n_pan))

        # directions and vectors (canonical frame)
        dirs = az_el_dirs(az, elev)                       # (n_tilt, n_pan, 3)
        vecs = dirs * range_img[..., None]                # (n_tilt, n_pan, 3)

        return {util.DEFAULT_OUTPUT_SLOT: vecs}

    
