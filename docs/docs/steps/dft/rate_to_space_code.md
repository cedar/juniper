# RateToSpaceCode

```python
RateToSpaceCode(name: str, shape: tuple[int, ...], limits: tuple[int, ...], center: tuple[int, ...]=_center, amplitude: float=_amplitude, sigma: tuple[int, ...]=_sigma, cyclic: bool=_cyclic)
```

## Description
Takes a vector and produces a Gaussian centered at corresponding field coordinates.

Note: Implement cyclic mode.

## Parameters-
- shape : tuple(Nx,Ny,...)
- limits : tuple((lx,ux), (ly,uy), ...)
- center (optional): tuple(x,y,z)
    - Default = tuple((ux+lx)/2, (uy+ly)/2, ...)
- amplitude (optional) : float
    - Default = 1.0
- sigma (optional) : tuple(sx,sy,...)
    - Default = (1.0,1.0,...)
- cyclic (optional) : bool
    - Default = False

## Slots--
- in0 : jnp.ndarray(len(shape))
- out0 : jnp.ndarray(shape)

## Import

```python
from juniper import RateToSpaceCode
```
