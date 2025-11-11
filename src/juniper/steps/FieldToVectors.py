from .Step import Step
from functools import partial
from .. import util
import jax.numpy as jnp
import jax

class FieldToVectors(Step):
    """
    Converts a 3D field into a set of 3D vectors.
    
    Parameters
    ----------
    - origin [m] (optional) : tuple(ox,oy,oz)
        - Origin of field with respect to origin of vector space. (Default: (0,0,0))
    - field_units_per_meter [1/m] (optional) : tuple(dx,dy,dz)
        - Number of field bins per meter. (Default: (100,100,100))
    - threshold (optional) : float
        - Only field units pircing the threshold are converted to vectors, all other are mapped to 0. (Default: 0.9)
    """
    def __init__(self, name, params):
        mandatory_params = []
        super().__init__(name, params, mandatory_params)

        if "origin" not in self._params.keys():
            self._params["origin"] = (0.0,0.0,0.0)

        if "field_units_per_meter" not in self._params.keys():
            self._params["field_units_per_meter"] = (100.0,100.0,100.0)

        if "threshold" not in self._params.keys():
            self._params["threshold"] = 0.9

    
    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):

        field = jnp.asarray(input_mats[util.DEFAULT_INPUT_SLOT], dtype=jnp.float32)
        Nx, Ny, Nz = field.shape
        ox, oy, oz = map(float, self._params["origin"])
        dx, dy, dz = map(float, self._params["field_units_per_meter"])

        mask = field > self._params["threshold"]
        count = jnp.sum(mask, dtype=jnp.int32)        # how many valid voxels

        # Get ALL indices in a fixed-size (padded) way
        # nonzero(..., size=...) guarantees a static-length result under jit.
        size = field.size
        ix, iy, iz = jnp.nonzero(mask, size=size, fill_value=0)  # each (size,)

        # Convert voxel indices -> world coordinates at voxel centers
        ix_f = ix.astype(jnp.float32)
        iy_f = iy.astype(jnp.float32)
        iz_f = iz.astype(jnp.float32)
        x = ox + (ix_f + 0.5) / dx
        y = oy + (iy_f + 0.5) / dy
        z = oz + (iz_f + 0.5) / dz
        coords_all = jnp.stack([x, y, z], axis=1)     # shape (size, 3)

        return {util.DEFAULT_OUTPUT_SLOT: coords_all}

    
