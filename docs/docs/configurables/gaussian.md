# Gaussian

```python
Gaussian(params)
```

## Description
A wrapper object for production of nd-gaussians. Can be used as a kernel for neural fields or convolutions. This class is also used by GaussianInput and other steps
to construct their output. If the Gaussian should be of a specific shape the 'max_shape' and 'shape' parameters should be specified otherwise the shape of the Gaussian
will be infered from the sigma parameter. Per default the Gaussian will be factorized per dimension, to be used for efficiant convolution. To materialize the full kernel,
set the factorized parameter to False.

## Parameters-
- sigma : tuple(s1,s2,...)
- amplitude : float
- normalized : bool
- shape (optional) : tuple(Nx,Ny,...)
- max_shape (optional) : tuple(Mx,My,...)
- factorized (optional) : bool
    - Default = True

## Import

```python
from juniper import Gaussian
```
