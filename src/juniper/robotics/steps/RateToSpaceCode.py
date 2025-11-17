from ...configurables.Step import Step
from functools import partial
from ...util import util
from ...configurables.Gaussian import Gaussian
import jax.numpy as jnp
import jax

class RateToSpaceCode(Step):
    """
    Description
    ---------
    Takes a vector and produces a Gaussian centered at corresponding field coordinates.

    TODO: Implement cyclic mode.
    
    Parameters
    ----------
    - shape : tuple(Nx,Ny,...)
    - limits : tuple((lx,ux), (ly,uy), ...)
    - center (optional): tuple(x,y,z)
        - Default = tuple((ux+lx)/2, (uy+ly)/2, ...)
    - amplitude (optional) : float
        - Default = 1.0
    - sigma (optional) : tuple(sx,sy,...)
        - Default = (1.0,1.0,...)
    - cyclic (optional) : bool
        - Default = False

    Step Input/Output slots
    -----------
    - in0 : jnp.ndarray(len(shape))
    - out0 : jnp.ndarray(shape)
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ['shape', 'limits']
        super().__init__(name, params, mandatory_params)

        if "center" not in self._params.keys():
            limits = self._params["limits"]
            self._params["center"] = [(limits[i][1]+limits[i][0])/2 for i in range(len(limits))]

        if "amplitude" not in self._params.keys():
            self._params["amplitude"] = 1.0

        if "sigma" not in self._params.keys():
            self._params["sigma"] = tuple([1.0 for i in range(len(self._params["shape"]))])

        if "cyclic" not in self._params.keys():
            self._params["cyclic"] = False

        self._limits = jnp.asarray(self._params["limits"], dtype=jnp.float32)
        self._intveral_sizes = self._limits[:,1] - self._limits[:,0]

        # pre generate gaussians
        self._gaussian = Gaussian(params={"sigma": self._params["sigma"], "amplitude": 0, "center": [x // 2 for x in self._params["shape"]], "shape": self._params["shape"], 
                                  "normalized": False, "factorized": False})
        
        self.reset()
        
        
    def reset(self):
        self.buffer[util.DEFAULT_INPUT_SLOT] = jnp.zeros((len(self._params["shape"]),))
    
    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input_vec = jnp.asarray(input_mats[util.DEFAULT_INPUT_SLOT], dtype=jnp.float32)

        # calculate centers
        center = (input_vec - self._limits[:,0]) - jnp.asarray(self._params["center"], dtype=jnp.float32)

        # update gaussian centers
        self._gaussian._params["center"] = center
        self._gaussian._params["amplitude"] = self._params["amplitude"] * (jnp.sum(center)>0)
        self._gaussian._gaussian = self._gaussian.gen_gauss_for_all_shapes(normalize=False, factorize=False)
        

        return {util.DEFAULT_OUTPUT_SLOT: self._gaussian.get_kernel()}

    
