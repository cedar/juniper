from ..configurables.Step import Step
from ..configurables.Gaussian import Gaussian
from ..util import util
import warnings
import jax.numpy as jnp

def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        return {util.DEFAULT_OUTPUT_SLOT: buffer["output"], "output":buffer["output"]}
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



        self.gaussian = Gaussian({"shape": params["shape"], "sigma": params["sigma"], "amplitude": params["amplitude"], "normalized": False, "center": params["center"], "factorized": False})
        self.register_buffer("output")
        self.buffer["output"] = self.gaussian.get_kernel()
        self.compute_kernel = compute_kernel_factory(self._params)
    
    def reset(self):
        reset_state = {}
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self._kernel
        reset_state["output"] = self.gaussian.get_kernel()
        return reset_state
    
    def set_data(self, gaussian):
        self.gaussian = gaussian

    def get_data(self):
        return self.gaussian.get_kernel()