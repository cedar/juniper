from .Step import Step
from ..GaussKernel import GaussKernel
from .. import util
import jax
from functools import partial
import warnings

class GaussInput(Step):

    def __init__(self, name, params):
        mandatory_params = ["shape", "sigma", "amplitude"]
        super().__init__(name, params, mandatory_params)

        if len(params["shape"]) != len(params["sigma"]):
            raise ValueError(f"GaussInput {name} requires equal dimensionality of sigma ({len(params['sigma'])}) and shape ({len(params['shape'])})")

        # Check if a center for the gaussian is given, otherwise default to (0, 0) (center of the shape)
        if "center" not in params:
            warnings.warn(f"GaussInput {name} does not have a center parameter. Defaulting to (0, 0).")
            params["center"] = (0,) * len(params["shape"])
        
        self.is_source = True
        
        # Remove default input slot
        self.input_slot_names = []
        self._max_incoming_connections = {}

        # Compute kernel once and save it
        kernel = GaussKernel({"shape": params["shape"], "sigma": params["sigma"], "amplitude": params["amplitude"], "normalized": False, "center": params["center"]})
        self._kernel = kernel.get_kernel()

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        
        output = self._kernel

        return {util.DEFAULT_OUTPUT_SLOT: output}