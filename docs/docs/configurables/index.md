# Configurables

Configurables are parameterizable objects used to construct kernels, sigmoids, and other components that are passed as parameters to steps. They are not steps themselves -- they don't participate in the computational graph directly, but are used to configure steps at initialization.

All configurables inherit from the `Configurable` base class, which handles parameter validation.

| Configurable | Description |
|--------------|-------------|
| [Gaussian](gaussian.md) | N-dimensional Gaussian kernel |
| [LateralKernel](lateral_kernel.md) | Combined lateral interaction kernel from multiple Gaussians |
| [Sigmoid](sigmoid.md) | Wrapper for sigmoid activation functions |
