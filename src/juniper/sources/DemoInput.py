from ..configurables.Step import Step
from ..configurables.Gaussian import Gaussian
from ..util import util
import warnings
import jax.numpy as jnp

def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):

        gaussian = Gaussian({"shape": params["shape"], "sigma": params["sigma"], "amplitude": buffer["gauss_params"][1], "normalized": False, "center": buffer["gauss_params"][0], "factorized": False})
        kernel = gaussian.get_kernel()
        return {util.DEFAULT_OUTPUT_SLOT: kernel, "gauss_params":buffer["gauss_params"]}
    return compute_kernel

class DemoInput(Step):
    """
    Description
    ---------
    DemoInput is a GaussInput that can be customized during runtime.

    Parameters
    ---------
    - shape : tuple((Nx,Ny,...))
    - sigma : float
    - amplitude : amplitude

    Step Input/Output slots
    ---------
    - out0 : jnp.array(shape)
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["shape", "sigma", "amplitude"]
        super().__init__(name, params, mandatory_params)

        if len(params["shape"]) != len(params["sigma"]):
            raise ValueError(f"DemoInput {name} requires equal dimensionality of sigma ({len(params['sigma'])}) and shape ({len(params['shape'])})")

        # Check if a center for the gaussian is given, otherwise default to (0, 0) (center of the shape)
        if "center" not in params:
            warnings.warn(f"DemoInput {name} does not have a center parameter. Defaulting to (0, 0).")
            params["center"] = (0,) * len(params["shape"])
        
        self.is_source = True
        
        # Remove default input slot
        self.input_slot_names = []
        self._max_incoming_connections = {}



        kernel = Gaussian({"shape": params["shape"], "sigma": params["sigma"], "amplitude": params["amplitude"], "normalized": False, "center": params["center"], "factorized": False})
        self._kernel = kernel.get_kernel()
        self.register_buffer("gauss_params")
        self.editable_gauss_params = jnp.zeros((2,))
        self.editable_gauss_params = self.editable_gauss_params.at[0].set(self._params["center"])
        self.editable_gauss_params = self.editable_gauss_params.at[1].set(self._params["amplitude"])
        self.buffer["gauss_params"] = self.editable_gauss_params
        
        self.compute_kernel = compute_kernel_factory(self._params)
    
    def reset(self):
        reset_state = {}
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self._kernel
        reset_state["gauss_params"] = self.editable_gauss_params
        return reset_state