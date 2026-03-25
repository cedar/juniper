from ...configurables.Configurable import Configurable
import jax.numpy as jnp

def compute_kernel_factory(params, M_func):
    def compute_kernel(input_mats, joint_state):
        # input_mats shape [N,3]
        # needed calc shape [4,N]
        # output shape [N,3]
        out_vec = jnp.ones((input_mats.shape[0], 4))
        out_vec = out_vec.at[:,:3].set(input_mats)
        out_vec = M_func(joint_state) @ out_vec.T
        return out_vec[:3].T
    return compute_kernel

class Transform(Configurable):
    """
    Description
    ---------
    A wrapper object for a rigid coordinate transformation. The coordinate transformation is given as a function that 
    parametarizes the transformation according to the kinematics of the space (i.e. joint states). The function should 
    return a 4x4 rigid transformation matrix for any given joint state. The compute function takes a set of vectors and
    a joint state to transform the vectors into a new coordinate frame.

    This object is used to construct a frame graph, which can be used by the CoordinateTransformation step.

    Parameters
    ----------
    - M_func: function object 
        - eg. lambda joint_state: jnp.eye(4) (identity)

    compute
    ----------
    - input_vec : jnp.ndarray((N,3))
    - joint_state : jnp.ndarray
    - out : jnp.ndarray((N,3))
    """
    def __init__(self, params):
        mandatory_params = ["M_func"]
        super().__init__(params, mandatory_params)

        self.M_func = params["M_func"]

        self.compute_kernel = compute_kernel_factory(self._params, self.M_func)

    def get_transform(self):
        return self.M_func
    