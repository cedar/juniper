from ..configurables.Step import Step
from ..configurables.Gaussian import Gaussian
from ..util import util
import jax
from functools import partial
import warnings

def compute_kernel_factory(kernel):
    return lambda input_mats, buffer, **kwargs: {util.DEFAULT_OUTPUT_SLOT: kernel}

class GaussInput(Step):
    """
    Description
    ---------
    Gaussian Input.

    Parameters
    ---------
    - shape : tuple((Nx,Ny,...))
    - sigma : tuple((sx,sy,...))
    - amplitude : float

    Step Input/Output slots
    ---------
    - in0: jnp.array(shape)
    - out0: jnp.array(shape)
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["shape", "sigma", "amplitude"]
        super().__init__(name, params, mandatory_params)

        if len(params["shape"]) != len(params["sigma"]):
            raise ValueError(f"GaussInput {name} requires equal dimensionality of sigma ({len(params['sigma'])}) and shape ({len(params['shape'])})")

        # Check if a center for the gaussian is given, otherwise default to (0, 0) (center of the shape)
        if "center" not in params:
            warnings.warn(f"GaussInput {name} does not have a center parameter. Defaulting to (0, 0).")
            self._params["center"] = [x // 2 for x in self._params["shape"]]
        
        self.is_source = True
        
        # Remove default input slot
        self.input_slot_names = []
        self._max_incoming_connections = {}

        # Compute kernel once and save it
        kernel = Gaussian({"shape": params["shape"], "sigma": params["sigma"], "amplitude": params["amplitude"], "normalized": False, "center": params["center"], "factorized": False})
        self._kernel = kernel.get_kernel()
        self.compute_kernel = compute_kernel_factory(self._kernel)

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, buffer, **kwargs):
        return self.compute_kernel(input_mats, buffer, **kwargs)
