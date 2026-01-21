### ToDo find a better name for this file and class
import jax.numpy as jnp
from .Configurable import Configurable
from ..math.Convolution import convolve_func_singleton

class LateralKernel(Configurable):
    """
    Description
    ---------
    This class generates a lateral kernel object. Inputs should all be of the kernel type (include a get_kernel() method) and have the same dimensionality.
    if the dimension sizes are not the same, the kernel with the largest dim size will be used and the others will be padded with zeros.

    Parameters
    ----------
    - kernels : list([Gaussian])
    """
    def __init__(self, params):
        mandatory_params = ["kernels"]
        self._name = "lateral_kernel"
        # optional parameters: wheights
        super().__init__(params, mandatory_params)
        
        self._shapes = [kernel._params["shape"] for kernel in params["kernels"]]
        self._dimensionalities = [len(shape) for shape in self._shapes]
        self._input_kernels = [kernel.get_kernel() for kernel in params["kernels"]]
        self.factorizations = [kernel._params["factorized"] for kernel in params["kernels"]]

        if any(dim != self._dimensionalities[0] for dim in self._dimensionalities):
            raise ValueError("All kernels must have the same dimensionality.")
        
        if any(fact != self.factorizations[0] for fact in self.factorizations):
            raise ValueError("All kernels must either be factorized or full.")
        self.factorized = self.factorizations[0]

        self._kernel = self._input_kernels if self.factorized else self.gen_kernel(self._input_kernels, self._shapes)
        
    def pad_kernels(self, kernels, shapes):
        if self.factorized:
            # kernels: List[List[jnp.ndarray]]  -> each kernel is a list of 1D factors (one per dimension)
            # shapes:  List[List[int]]          -> each kernel's per-dimension sizes (same order as factors)

            # determine the target size for each dimension across all kernels
            num_dims = len(shapes[0])
            target_sizes = [max(s[d] for s in shapes) for d in range(num_dims)]

            # pad each factor (dimension) of each kernel to that dimension's target size
            padded_kernels = []
            for k_factors, k_shapes in zip(kernels, shapes):
                padded_factors = []
                for d, (factor, size) in enumerate(zip(k_factors, k_shapes)):
                    diff = target_sizes[d] - size
                    left = diff // 2
                    right = diff - left  # puts the extra one on the right if diff is odd
                    # factors are 1D; pad along their single axis
                    pad_spec = [(left, right)]
                    padded_factors.append(jnp.pad(factor, pad_spec, mode='constant', constant_values=0))
                padded_kernels.append(padded_factors)
        else:
            # determine the maximum shape for each dimension
            target_shape = []
            for dim in range(len(shapes[0])):
                max_size = max(shape[dim] for shape in shapes)
                target_shape.append(max_size)

            # pad each kernel to the target shape
            padded_kernels = []
            for kernel, shape in zip(kernels, shapes):
                padding = [((target_shape[dim]-shape[dim]) // 2, (target_shape[dim]-shape[dim]) // 2) for dim in range(len(shape))]
                padded_kernels.append(jnp.pad(kernel, padding, mode='constant', constant_values=0))

        return padded_kernels
    
    def gen_kernel(self, kernels, shapes):
        # combine the kernels into a single kernel by summing them up
        padded_kernels = self.pad_kernels(kernels, shapes)
        combined_kernel = None
        for kernel in padded_kernels:
            if combined_kernel is None:
                combined_kernel = kernel
            else:
                if self.factorized:
                    for d in range(len(kernel)):
                        combined_kernel[d] += kernel[d]
                else:
                    combined_kernel += kernel
        return combined_kernel

    def get_kernel(self):
        return self._kernel
    
    def gen_convolve_func(self):
        self.convolve = convolve_func_singleton(self._kernel, self.factorized)
        return self.convolve
    