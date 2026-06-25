import logging
from ..core.frontend.Source import Source
from ..math.Gaussian import Gaussian
from ..util import util
import warnings
from ..core.backend.Exceptions import JuniperConfigurationError
from ..core.backend.Warnings import JuniperConfigurationWarning


logger = logging.getLogger(__name__)
def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        return {util.DEFAULT_OUTPUT_SLOT: buffer[util.DEFAULT_OUTPUT_SLOT]}
    return compute_kernel

class DemoInput(Source):
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
    _center = None
    def __init__(self, name : str, shape : tuple, sigma : tuple, amplitude : float, center = _center):
        params = locals().copy()
        mandatory_params = ["shape", "sigma", "amplitude"]
        super().__init__(name, params, mandatory_params)

        if len(shape) != len(sigma):
            raise JuniperConfigurationError(f"DemoInput {name} requires equal dimensionality of sigma ({len(sigma)}) and shape ({len(shape)})")

        # Check if a center for the gaussian is given, otherwise default to (0, 0) (center of the shape)
        if center is None:
            warnings.warn(f"DemoInput '{self.get_path_str()}' does not have a center parameter. Defaulting to (0, 0).", JuniperConfigurationWarning, stacklevel=2)
            self._params["center"] = (0,) * len(shape)
        
        # Remove default input slot
        self.input_slot_names = []



        self.gaussian = Gaussian({"shape": self._params["shape"], "sigma": self._params["sigma"], "amplitude": self._params["amplitude"], "normalized": False, "center": self._params["center"], "factorized": False})
        self.compute_kernel = compute_kernel_factory(self._params)
    
    def set_data(self, gaussian):
        self.gaussian = gaussian

    def get_data(self):
        return self.gaussian.get_kernel()
