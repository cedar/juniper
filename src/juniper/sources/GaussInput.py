import logging
from ..core.frontend.Source import Source
from ..math.Gaussian import Gaussian
from ..util import util
from ..util import util_jax
import warnings
from ..core.backend.Exceptions import JuniperConfigurationError
from ..core.backend.Warnings import JuniperConfigurationWarning
import jax.numpy as jnp


logger = logging.getLogger(__name__)
def compute_kernel_factory(kernel):
    kernel = jnp.asarray(kernel, dtype=util_jax.cfg["jdtype"])
    return lambda input_mats, buffer, **kwargs: {util.DEFAULT_OUTPUT_SLOT: kernel}

class GaussInput(Source):
    """
    Description
    ---------
    Gaussian Input.

    Parameters
    ---------
    - shape : tuple((Nx,Ny,...))
    - sigma : tuple((sx,sy,...))
    - amplitude : float
    - center(optional) : tuple((cx,cy,...))
        - Center of the gaussian. (Default: (Nx/2,Ny/2,...))

    Step Input/Output slots
    ---------
    - in0: jnp.array(shape)
    - out0: jnp.array(shape)
    """
    _center = None
    def __init__(self, name : str, shape : tuple, sigma : tuple, amplitude : float, center : tuple | None = _center):
        params = locals().copy()
        mandatory_params = ["shape", "sigma", "amplitude"]
        super().__init__(name, params, mandatory_params)

        if len(shape) != len(sigma):
            raise JuniperConfigurationError(f"GaussInput {name} requires equal dimensionality of sigma ({len(sigma)}) and shape ({len(shape)})")

        # Check if a center for the gaussian is given, otherwise default to (0, 0) (center of the shape)
        if center is None:
            warnings.warn(f"GaussInput {name} does not have a center parameter.", JuniperConfigurationWarning, stacklevel=2)
            self._params["center"] = [x // 2 for x in self._params["shape"]]

        # Compute kernel once and save it
        kernel = Gaussian({"shape": self._params["shape"], "sigma": self._params["sigma"], "amplitude": self._params["amplitude"], "normalized": False, "center": self._params["center"], "factorized": False})
        self._kernel = kernel.get_kernel()
        self.compute_kernel = compute_kernel_factory(self._kernel)

    def get_data(self):
        pass

    def infer_output_shapes(self, input_specs):
        return {util.DEFAULT_OUTPUT_SLOT: self._params["shape"]}