### ToDo find a better name for this file and class
import jax.numpy as jnp
import jax
from src import util_jax
from src.Configurable import Configurable

class LinearKernelCombination(Configurable):
    # This class generates a wheighted linear combination of a list of input kernels. Inputs should all be of the kernel type (include a get_kernel() method) and have the same dimensionality.
    # if the dimension sizes are not the same, the kernel with the largest dim size will be used and the others will be padded with zeros.
    def __init__(self, params):
        mandatory_params = ["kernels"]
        # optional parameters: wheights
        super().__init__(params, mandatory_params)
        self._shapes = [kernel.get_kernel().shape for kernel in params["kernels"]]
        self._dimensionalities = [len(shape) for shape in self._shapes]
        self._input_kernels = [kernel.get_kernel() for kernel in params["kernels"]]
        
        if not "wheights" in params:
            params["wheights"] = [1.0 for _ in range(len(params["kernels"]))]

        if any(dim != self._dimensionalities[0] for dim in self._dimensionalities):
            raise ValueError("All kernels must have the same dimensionality.")
        
        self._input_kernels = self.pad_kernels(self._input_kernels, self._shapes)
        
        # materialize the kernel tensor
        self._kernel = None
        for kernel, wheight in zip(self._input_kernels, params["wheights"]):
            if self._kernel is None:
                self._kernel = kernel * wheight
            else:
                self._kernel += kernel * wheight
        
        del self._input_kernels # delete the input kernels to save memory, we only need the combined kernel
    

    def pad_kernels(self, kernels, shapes):
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
    
    def get_kernel(self):
        return self._kernel
    