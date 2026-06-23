import logging
from ...core.frontend.Step import Step
from ...util import util
import jax.numpy as jnp
import numpy as np



logger = logging.getLogger(__name__)
class FieldToPointCloud(Step):
    """
    Description
    ---------
    Converts a 3D field into a Point Cloud.
    
    Parameters
    ----------
    - origin [m] (optional) : tuple(ox,oy,oz)
        - Origin of field with respect to origin of vector space. 
        - Default = (0,0,0)
    - field_units_per_meter [1/m] (optional) : tuple(dx,dy,dz)
        - Number of field bins per meter. 
        - Default = (100,100,100)
    - threshold (optional) : float
        - Only field units pircing the threshold are converted to vectors, all other are mapped to 0. 
        - Default = 0.9
    - N_pt (optional) : int
        - The number of points in the cloud. 0-vectors are returned if less field points pierce the threshold.
        - Default = FieldSize

    Step Input/Output slots
    ---------
    - in0 : jnp.array((Nx,Ny,Nz))
    - out0 : jnp.array((Nx*Ny*Nz,3))
    """
    _origin = (0.0,0.0,0.0)
    _field_units_per_meter = (100.0,100.0,100.0)
    _threshold = 0.9
    _N_pt = jnp.inf
    def __init__(
            self,
            name : str,
            origin : tuple = _origin,
            field_units_per_meter : tuple = _field_units_per_meter,
            threshold : float = _threshold,
            N_pt = _N_pt):
        params = locals().copy()
        mandatory_params = []
        super().__init__(name, params, mandatory_params)

        self.compute_kernel = compute_kernel_factory(self._params)

    def infer_output_shapes(self, input_specs):
        if util.DEFAULT_INPUT_SLOT not in input_specs:
            return {}
        shape = input_specs[util.DEFAULT_INPUT_SLOT][0]
        n_pt = self._params["N_pt"]
        if n_pt == jnp.inf:
            n_pt = int(np.prod(shape)) if "np" in globals() else shape[0] * shape[1] * shape[2]
        return {util.DEFAULT_OUTPUT_SLOT: (int(n_pt), 3)}

    
def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):

        field = jnp.asarray(input_mats[util.DEFAULT_INPUT_SLOT], dtype=jnp.float32)
        Nx, Ny, Nz = field.shape
        ox, oy, oz = map(float, params["origin"])
        dx, dy, dz = map(float, params["field_units_per_meter"])

        mask = field > params["threshold"]
        count = jnp.sum(mask, dtype=jnp.int32)        # how many valid voxels  # noqa: F841

        # Get ALL indices in a fixed-size (padded) way
        # nonzero(..., size=...) guarantees a static-length result under jit.
        size = min(field.size,params["N_pt"])
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
    return compute_kernel

    
