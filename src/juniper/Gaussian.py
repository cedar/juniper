import jax.numpy as jnp
from . import util_jax
from .Configurable import Configurable
from .Convolution import convolve_func_singleton

class Gaussian(Configurable):
    """
    Description
    ---------
    A wrapper object for production of nd-gaussians. Can be used as a kernel for neural fields or convolutions. This class is also used by GaussianInput and other steps
    to construct their output. If the Gaussian should be of a specific shape the 'max_shape' and 'shape' parameters should be specified otherwise the shape of the Gaussian
    will be infered from the sigma parameter. Per default the Gaussian will be factorized per dimension, to be used for efficiant convolution. To materialize the full kernel,
    set the factorized parameter to False.

    Parameters
    ----------
    - sigma : tuple(s1,s2,...)
    - amplitude : float
    - normalized : bool
    - shape (optional) : tuple(Nx,Ny,...)
    - max_shape (optional) : tuple(Mx,My,...)
    - factorized (optional) : bool
        - Default = True

    """
    def __init__(self, params):
        mandatory_params = ["sigma", "amplitude", "normalized"]
        super().__init__(params, mandatory_params)
        self._dimensionality = len(params["sigma"])
        # Estimate width if not explicitly set
        if "shape" not in params:
            self._params["shape"] = self._estimate_size()

        # Center of the kernel, default is the center of the shape
        if "center" not in params:
            self._params["center"] = [x // 2 for x in self._params["shape"]]

        # per default we use a factorized kernel
        if "factorized" not in params:
            self._params["factorized"] = True

        # materialize the kernel tensor
        self._gaussian = self.gen_gauss_for_all_shapes(self._params["normalized"], self._params["factorized"])

        # check if the kernel size is within the max_shape
        if "max_shape" in params:
            self.check_gaussian_size(self._params["factorized"])

        if len(params["shape"]) != len(params["sigma"]):
            raise ValueError(f"Gaussian requires equal dimensionality of sigma ({len(params['sigma'])}) and shape ({len(params['shape'])})")

    def _estimate_size(self):
        limit = 5
        widths = []
        for dim in range(self._dimensionality):
            sigma = self._params["sigma"][dim]
            if sigma == 0:
                widths.append(1)
            else:
                width = int(jnp.ceil(limit * sigma))
                if width % 2 == 0:
                    width += 1
                widths.append(width)
        return widths
    
    def check_gaussian_size(self, factorize):
        gaussian_size = self._params["shape"]
        max_shape = self._params["max_shape"]
        if len(gaussian_size) != len(max_shape):
            raise Exception(f"Dimensionality of kernel size and max_shape must be equal ({len(gaussian_size)} != {len(max_shape)})")

        for dim in range(len(gaussian_size)):
            if gaussian_size[dim] > max_shape[dim]:
                # Trim the kernel in the current dimension to be max_shape[dim] wide (or -1 if not odd)
                new_size = max_shape[dim] - (1 if max_shape[dim] % 2 == 0 else 0)
                #self._kernel = self._kernel[tuple(slice(None) if i != dim else slice(new_size) for i in range(len(kernel_size)))]
                # Trim it so that the center of the kernel is at the center of the max_shape
                center = gaussian_size[dim] // 2
                new_center = new_size // 2
                start = center - new_center
                end = start + new_size
                slices = [slice(None) if i != dim else slice(start, end) for i in range(len(gaussian_size))]
                self._params["shape"][dim] = new_size
                if factorize:
                    self._gaussian[dim] = self._gaussian[dim][start:end]
                else:
                    self._gaussian = self._gaussian[tuple(slices)]

    def gen_gauss_for_all_shapes(self, normalize, factorize):
        shape = self._params["shape"]
        center = self._params["center"]
        sigma = self._params["sigma"]
        sign = jnp.sign(self._params["amplitude"])
        gaussian = [] if factorize else 1.0
        for dim, s in enumerate(shape):
            gauss_1d = jnp.exp(-0.5 * (jnp.square(jnp.arange(s, dtype=util_jax.cfg["jdtype"]) - center[dim]) / jnp.square(sigma[dim])))
            if normalize:
                gauss_1d /= jnp.sum(gauss_1d)
            gauss_1d *= jnp.abs(self._params["amplitude"])**(1/self._dimensionality)
            if factorize:
                gaussian += [gauss_1d]
            else:
                shape_reshape = [1]*len(shape)
                shape_reshape[dim] = s
                gaussian = gaussian * gauss_1d.reshape(shape_reshape) 
        if factorize:
            gaussian[0] *= sign
        else:
            gaussian *= sign
        return gaussian
    
    def get_kernel(self):
        return self._gaussian
    
    def gen_convolve_func(self):
        self.convolve = convolve_func_singleton([self._gaussian], self._params["factorized"])
        return self.convolve
    