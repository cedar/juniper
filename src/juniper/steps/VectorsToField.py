from .Step import Step
from functools import partial
from .. import util
import jax.numpy as jnp
import jax

import jax.debug as jgdb
class VectorsToField(Step):
    """
    Converts 3D Cartesian vectors into a 3D field like representation (Nx, Ny, Nz).
    
    Parameters
    ----------
    - field_shape : tuple(Nx,Ny,Nz)
    - origin [m] (optional) : tuple(ox,oy,oz)
        - Origin of field with respect to origin of vector space. (Default: (0,0,0))
    - field_units_per_meter [1/m] (optional) : tuple(dx,dy,dz)
        - Number of field bins per meter. (Default: (100,100,100))

    """
    def __init__(self, name, params):
        mandatory_params = ["field_shape"]
        super().__init__(name, params, mandatory_params)

        if "origin" not in self._params.keys():
            self._params["origin"] = (0.0,0.0,0.0)

        if "field_units_per_meter" not in self._params.keys():
            self._params["field_units_per_meter"] = (100.0,100.0,100.0)

    
    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):

        v = jnp.asarray(input_mats[util.DEFAULT_INPUT_SLOT], dtype=jnp.float32)

        Nx, Ny, Nz = map(int, self._params["field_shape"])
        ox, oy, oz = map(float, self._params["origin"])
        dx, dy, dz = map(float, self._params["field_units_per_meter"])

        # world -> voxel indices
        ix = jnp.floor((v[:, 0] - ox) * dx).astype(jnp.int32)
        iy = jnp.floor((v[:, 1] - oy) * dy).astype(jnp.int32)
        iz = jnp.floor((v[:, 2] - oz) * dz).astype(jnp.int32)
        jgdb.print('x_min,x_max: {}, {}', jnp.min(ix), jnp.max(ix))
        jgdb.print('y_min,y_max: {}, {}', jnp.min(iy), jnp.max(iy))
        jgdb.print('z_min,z_max: {}, {}', jnp.min(iz), jnp.max(iz))
        jgdb.print('{}', self._params)

        # in-bounds mask (no boolean indexing!)
        inb = (
            (ix >= 0) & (ix < Nx) &
            (iy >= 0) & (iy < Ny) &
            (iz >= 0) & (iz < Nz)
        )

        # Redirect OOB indices to a safe in-bounds voxel and zero-out their values.
        # Then use a reduction update (max) so zeros can't clobber ones.
        safe_ix = jnp.where(inb, ix, 0)
        safe_iy = jnp.where(inb, iy, 0)
        safe_iz = jnp.where(inb, iz, 0)
        vals    = inb.astype(jnp.float32)  # 1 for in-bounds, 0 for OOB

        field = jnp.zeros((Nx, Ny, Nz), dtype=jnp.float32)
        field = field.at[safe_ix, safe_iy, safe_iz].max(vals)  # robust to duplicates

        return {util.DEFAULT_OUTPUT_SLOT: field}

    
