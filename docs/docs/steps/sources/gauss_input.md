# GaussInput

```python
GaussInput(name: str, shape: tuple, sigma: tuple, amplitude: float, center=_center)
```

## Description
Gaussian Input.

## Parameters
- shape : tuple((Nx,Ny,...))
- sigma : tuple((sx,sy,...))
- amplitude : float
- center(optional) : tuple((cx,cy,...))
    - Center of the gaussian. (Default: (Nx/2,Ny/2,...))

## Slots
- in0: jnp.array(shape)
- out0: jnp.array(shape)

## Import

```python
from juniper import GaussInput
```
