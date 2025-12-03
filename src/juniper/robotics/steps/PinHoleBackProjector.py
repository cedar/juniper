from ...configurables.Step import Step
from functools import partial
from ...util import util
import jax.numpy as jnp
import jax

def compute_kernel_factory(params, M_inv):
    def compute_kernel(input_mats, buffer, **kwargs):
        depth = input_mats[util.DEFAULT_INPUT_SLOT]
        H, W = depth.shape
        ys, xs = jnp.indices((H, W), dtype=jnp.float32)  # ys=row, xs=col
        # Build pixel homogeneous (x,y,1). Note: x=cols, y=rows
        pix = jnp.stack([xs, ys, jnp.ones_like(xs)], axis=0)  # (3, H, W)
        pix = pix.reshape(3, -1)  # (3, HW)

        # Rays in homogeneous 4-vector after pinv
        rays4 = M_inv @ pix  # (4, HW)
        rays3 = rays4[:3, :]

        # Normalize rays, then scale by per-pixel depth
        rays_norm = jnp.linalg.norm(rays3, axis=0, keepdims=True) + 1e-8
        unit_rays = rays3 / rays_norm                      # (3, HW)
        depth_flat = depth.reshape(-1)                     # (HW,)
        output = unit_rays * depth_flat                    # (3, HW)
        output = jnp.stack([output[2], output[0], output[1]])
        output = output.T                                   # (HW, 3)

        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel

class PinHoleBackProjector(Step):
    """
    Description
    ---------
    Takes the depth image of a pinhole camera as input and transforms the image into polar coordinates.

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
        self.M_cam_inv = jnp.linalg.pinv(self.M_cam)

        self.compute_kernel = compute_kernel_factory(self._params, self.M_cam_inv)

    
