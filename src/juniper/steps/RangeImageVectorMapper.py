from .Step import Step
from functools import partial
from .. import util
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

def range_image_to_vectors(range_img, img_shape, pan_range, tilt_range):
    """
    range_img: (n_tilt, n_pan) distances.
    img_shape: (n_tilt, n_pan) = (rows, cols)
    pan_range:  (pan_lo, pan_hi)   e.g. (-jnp.pi, jnp.pi)
    """
    n_tilt, n_pan = img_shape
    pan_lo, pan_hi = pan_range
    tilt_lo, tilt_hi = tilt_range

    # bin centers (uniform)
    pan_centers  = jnp.linspace(pan_lo,  pan_hi,  n_pan,  endpoint=False) + 0.5 * (pan_hi  - pan_lo ) / n_pan
    tilt_centers = jnp.linspace(tilt_lo, tilt_hi, n_tilt, endpoint=False) + 0.5 * (tilt_hi - tilt_lo) / n_tilt

    # per-pixel angles
    az   = jnp.broadcast_to(pan_centers,              (n_tilt, n_pan))
    elev = jnp.broadcast_to(tilt_centers[:, None],    (n_tilt, n_pan))

    # directions and vectors (canonical frame)
    dirs = az_el_dirs(az, elev)                       # (n_tilt, n_pan, 3)
    vecs = dirs * range_img[..., None]                # (n_tilt, n_pan, 3)


    return vecs  # (n_tilt, n_pan, 3)

def vectors_to_range_image(vectors, img_shape, pan_range, tilt_range, fill_value=0.0):
    """
    vectors: (N, 3) Cartesian (allocentric if origin!=0; we subtract it)
    fill_value: used for 'empty' bins (only for reduce='min')
    Returns: (n_tilt, n_pan) range image
    """
    v = jnp.asarray(vectors, dtype=jnp.float32)

    n_pan = img_shape[1]
    n_tilt = img_shape[0]
    pan_lo, pan_hi = pan_range
    tilt_lo, tilt_hi = tilt_range

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
    img = jnp.where(jnp.isinf(img), fill_value, img)

    return img

class RangeImageVectorMapper(Step):
    """
    Converts between spherical range images (tilt × pan) and 3D Cartesian vectors.

    Conventions:
      - pan (azimuth) ∈ [pan_lo, pan_hi], default [-pi, pi]
      - tilt (polar)  ∈ [tilt_lo, tilt_hi], default [0, pi]
      - range image shape: (n_tilt, n_pan)  [Y, X]
    """
    def __init__(self, name, params):
        mandatory_params = ["img_shape", "pan_range", "tilt_range", "output_type"]
        super().__init__(name, params, mandatory_params)

        self._params = params.copy()

        self.img_shape = self._params["img_shape"]
        self.pan_range = self._params["pan_range"]
        self.tilt_range = self._params["tilt_range"]
        
        if "image" == self._params["output_type"]:
            self.compute_func = vectors_to_range_image
        elif "vector" == self._params["output_type"]:
            self.compute_func = range_image_to_vectors

    
    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):

        output = self.compute_func(input_mats[util.DEFAULT_INPUT_SLOT], self.img_shape, self.pan_range, self.tilt_range)

        return {util.DEFAULT_OUTPUT_SLOT: output}

    
