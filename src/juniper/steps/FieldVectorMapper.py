from .Step import Step
from functools import partial
from .. import util
import jax.numpy as jnp
import jax

import jax.debug as jgdb

def vectors_to_field(vectors, field_shape):
    v = jnp.asarray(vectors, dtype=jnp.float32)

    Nx, Ny, Nz = map(int, field_shape)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = ((0,Nx),(0,Ny),(0,Nz))

    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    jgdb.print('x_min,x_max: {}, {}', jnp.min(x), jnp.max(x))
    jgdb.print('y_min,y_max: {}, {}', jnp.min(y), jnp.max(y))
    jgdb.print('z_min,z_max: {}, {}', jnp.min(z), jnp.max(z))
    eps = 1e-8
    ix = jnp.floor((x - xmin) / (xmax - xmin + eps) * Nx).astype(jnp.int32)
    iy = jnp.floor((y - ymin) / (ymax - ymin + eps) * Ny).astype(jnp.int32)
    iz = jnp.floor((z - zmin) / (zmax - zmin + eps) * Nz).astype(jnp.int32)
    ix = jnp.clip(ix, 0, Nx - 1)
    iy = jnp.clip(iy, 0, Ny - 1)
    iz = jnp.clip(iz, 0, Nz - 1)

    values = jnp.ones(v.shape[0], dtype=jnp.float32)
    field = jnp.zeros((Nx, Ny, Nz), dtype=jnp.float32)

    return field.at[ix,iy,iz].set(values)

def field_to_vectors(field, field_shape, threshold=0.9):
    field = jnp.asarray(field, dtype=jnp.float32)
    mask = field > threshold

    Nx, Ny, Nz = map(int, field_shape)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = ((0,Nx),(0,Ny),(0,Nz))

    # Create grid coordinates
    xs = jnp.linspace(xmin, xmax, Nx + 1)
    ys = jnp.linspace(ymin, ymax, Ny + 1)
    zs = jnp.linspace(zmin, zmax, Nz + 1)

    xs = 0.5 * (xs[:-1] + xs[1:])
    ys = 0.5 * (ys[:-1] + ys[1:])
    zs = 0.5 * (zs[:-1] + zs[1:])

    gx, gy, gz = jnp.meshgrid(xs, ys, zs, indexing="ij")
    coords = jnp.stack([gx[mask], gy[mask], gz[mask]], axis=1)

    return coords #(N,3)

class FieldVectorMapper(Step):
    """
    Converts between 3D Cartesian vectors and a voxel field (Nx, Ny, Nz).
    The field can be occupancy (0/1) or aggregate values (sum/max/min/set).
    """
    def __init__(self, name, params):
        """
        field_shape: (Nx, Ny, Nz)
        bounds: ((xmin,xmax), (ymin,ymax), (zmin,zmax)) for continuous coords
        """
        mandatory_params = ["field_shape", "output_type"]
        super().__init__(name, params, mandatory_params)

        self._params = params.copy()
        self.field_shape = self._params["field_shape"]

        if "vector" == self._params["output_type"]:
            self.compute_func = field_to_vectors
        elif "field" == self._params["output_type"]:
            self.compute_func = vectors_to_field
        else:
            raise(ValueError("Unrecognized output_type for FieldVectorMapper. Requires either field or vector."))

    
    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):

        output = self.compute_func(input_mats[util.DEFAULT_INPUT_SLOT], self.field_shape)

        return {util.DEFAULT_OUTPUT_SLOT: output}

    
