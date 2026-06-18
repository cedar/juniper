from ..core.frontend.Step import Step
from ..util import util
import jax.scipy as jsp

def compute_kernel_factory(params, kernel, use_dynamic):
    if use_dynamic:
        def compute_kernel(input_mats, buffer, **kwargs):
            
            # Input
            input_mat = input_mats[util.DEFAULT_INPUT_SLOT]

            k = input_mats["in1"]
            
            # Computation
            output = jsp.signal.fftconvolve(input_mat, k, mode=params["mode"])

            # Return output
            return {util.DEFAULT_OUTPUT_SLOT: output}
    else:
        def compute_kernel(input_mats, buffer, **kwargs):
            
            # Input
            input_mat = input_mats[util.DEFAULT_INPUT_SLOT]

            # Computation
            output = jsp.signal.fftconvolve(input_mat, kernel, mode=params["mode"])

            # Return output
            return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel

class Convolution(Step):
    """
    Description
    ---------
    Convolution of incoming step with kernel. The kernel can be given directly as a static kernel object (ie. Gaussian) or via an input connection as a dynamic kernel.

    Parameters
    ---------
    - kernel (optional) : LateralKernel
        - A dynamic kernel via input is used if this kernel is unspecified.
    - mode (optional) : str(same)
        - Default = same

    Step Input/Output slots
    ---------
    - in0 : jnp.array()
    - out0 : jnp.array()
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = []
        super().__init__(name, params, mandatory_params)
        self._use_dynamic = "kernel" not in self._params.keys()
        self._kernel = 0 if self._use_dynamic else self._params["kernel"].get_kernel()
        if "mode" not in self._params.keys():
            self._params["mode"] = "same"

        self._params["shape"] = (1,) # used for initial warmup to set input

        if self._use_dynamic:
            self.register_input_slot("in1")

        self.compute_kernel = compute_kernel_factory(self._params, self._kernel, self._use_dynamic)

    def infer_output_shapes(self, input_specs):
        if util.DEFAULT_INPUT_SLOT not in input_specs:
            return {}

        input_shape = tuple(input_specs[util.DEFAULT_INPUT_SLOT][0])
        mode = self._params["mode"]
        if mode == "same":
            return {util.DEFAULT_OUTPUT_SLOT: input_shape}

        kernel_shape = None
        if self._use_dynamic:
            if "in1" in input_specs:
                kernel_shape = tuple(input_specs["in1"][0])
        elif hasattr(self._kernel, "shape"):
            kernel_shape = tuple(self._kernel.shape)
        elif isinstance(self._kernel, (tuple, list)):
            kernel_shape = tuple(len(axis_kernel) for axis_kernel in self._kernel)

        if kernel_shape is None:
            return {}

        if mode == "full":
            output_shape = tuple(in_size + kernel_size - 1 for in_size, kernel_size in zip(input_shape, kernel_shape))
        elif mode == "valid":
            output_shape = tuple(abs(in_size - kernel_size) + 1 for in_size, kernel_size in zip(input_shape, kernel_shape))
        else:
            raise ValueError(f"Unknown convolution mode: {mode}")

        return {util.DEFAULT_OUTPUT_SLOT: output_shape}


    
