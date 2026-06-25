# Convolution

```python
Convolution(name: str, kernel=_kernel, mode: str=_mode)
```

## Description
Convolution of incoming step with kernel. The kernel can be given directly as a static kernel object (ie. Gaussian) or via an input connection as a dynamic kernel.

## Parameters
- kernel (optional) : LateralKernel
    - A dynamic kernel via input is used if this kernel is unspecified.
- mode (optional) : str(same)
    - Default = same

## Slots
- in0 : jnp.array()
- out0 : jnp.array()

## Import

```python
from juniper import Convolution
```

## Example

```python
conv = Convolution("conv", kernel=Gaussian({"shape": (21,), "sigma": (3,), "amplitude": 1.0, "normalized": True}))
# or omit kernel and connect a dynamic kernel to conv.in1
```
