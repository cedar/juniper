from ...configurables.Step import Step
from functools import partial
from ...util import util
import jax.numpy as jnp
import jax

def compute_kernel_factory(params, M):
    def compute_kernel(input_mats, buffer, **kwargs):
        # this is not quite the same as the backprojection.... 
        # the z-val needs to be tha val entry, not the vector norm
        # TODO make this exact inverse of backprojection
        point_cloud = input_mats[util.DEFAULT_INPUT_SLOT] # (N,3)
        point_cloud_reorder = point_cloud.copy()
        point_cloud = point_cloud.at[:,0].set(point_cloud_reorder[:,1])
        point_cloud = point_cloud.at[:,1].set(point_cloud_reorder[:,2])
        point_cloud = point_cloud.at[:,2].set(point_cloud_reorder[:,0])
        #vector_lenghts = jnp.linalg.norm(point_cloud, axis=1, keepdims=True) + 1e-8 # (N,1)
        vector_lenghts = jnp.array([point_cloud[:,2]]).T
        point_cloud = jnp.append(point_cloud, jnp.ones_like(vector_lenghts), axis=1) # (N,4)

        pix = M @ point_cloud.T # (3,N)
        pix = pix.at[:2,:].divide(pix[2,:]+1e-8)
        pix = jnp.round(pix).astype(jnp.int32)

        out = jnp.full(params["img_shape"], jnp.inf, dtype=jnp.float32) # (H,W)
        out = out.at[pix[1,:],pix[0,:]].min(vector_lenghts[:,0], mode="drop") # fill with pixel vals and take the closest value for each pixel
        out = jnp.where(jnp.isinf(out), 0, out) # replace inf values with 0

        return {util.DEFAULT_OUTPUT_SLOT: out}
    return compute_kernel

class PinHoleProjector(Step):
    """
    Description
    ---------
    Takes a point cloud and transforms it into a depth image of a pinhole depth camera.

    Parameters
    ---------    
    - img_shape : tuple(H,W)
    - focal_length : float
    - frustrum_angles : tuple(dphi, dtheta)

    Step Input/Output slots
    ---------
    - Input: jnp.array(img_shape)
    - output: jnp.ndarray(H*W,3)
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["img_shape", "focal_length", "frustrum_angles"]
        super().__init__(name, params, mandatory_params)
        self._params = params.copy()

        f = self._params["focal_length"]
        angleRangeX = self._params["frustrum_angles"][0] * jnp.pi / 180
        angleRangeY = self._params["frustrum_angles"][1] * jnp.pi / 180
        W = self._params["img_shape"][1]
        H = self._params["img_shape"][0]

        s_x = W / (2 * f * jnp.tan(angleRangeX / 2))
        s_y = H / (2 * f * jnp.tan(angleRangeY / 2))
        u_0 = W / 2
        v_0 = H / 2

        self.M_cam = jnp.zeros((3,4))
        self.M_cam = self.M_cam.at[0,0].set(s_x * f)
        self.M_cam = self.M_cam.at[0,2].set(u_0)
        self.M_cam = self.M_cam.at[1,1].set(s_y * f)
        self.M_cam = self.M_cam.at[1,2].set(v_0)
        self.M_cam = self.M_cam.at[2,2].set(1)

        self.compute_kernel = compute_kernel_factory(self._params, self.M_cam)

    
