from .Configurable import Configurable

import jax.numpy as jnp

class Transform(Configurable):
    def __init__(self, params):
        mandatory_params = ["M_func"]
        super().__init__(params, mandatory_params)

        self.M_func = params["M_func"]

    def get_transform(self):
        return self.M_func
    
    def compute(self, input_vec, joint_state):
        # input_vec shape [N,3]
        # needed calc shape [4,N]
        # output shape [N,3]
        out_vec = jnp.ones((input_vec.shape[0], 4))
        out_vec = out_vec.at[:,:3].set(input_vec)
        out_vec = self.M_func(joint_state) @ out_vec.T
        return out_vec[:3].T
    

"""class ComposedTransform(Transform):
    def __init__(self, params):
        mandatory_params = ["M_func"]
        super().__init__(params)

        self.M_func = params["M_func"] # list of funcs
    
    def compute(self, input_vec, joint_state):
        out = jnp.ones((input_vec.shape[0], 4))
        out.at[:,:3].set(input_vec)
        for f in self.M_func:
            out = f(joint_state) @ out
        return out[:3].T"""
        


        