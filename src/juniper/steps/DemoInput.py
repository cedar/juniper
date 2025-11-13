from .Step import Step
from ..Gaussian import Gaussian
from .. import util
import jax
from functools import partial
import warnings
import numpy as np

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

    def compute(self, input_mats, **kwargs):

        kernel = Gaussian({"shape": self._params["shape"], "sigma": self._params["sigma"], "amplitude": self._params["amplitude"], "normalized": False, "center": self._params["center"], "factorized": False})
        self._kernel = kernel.get_kernel()
        return {util.DEFAULT_OUTPUT_SLOT: self._kernel}