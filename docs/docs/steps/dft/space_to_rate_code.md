# SpaceToRateCode

```python
SpaceToRateCode(name: str, shape: tuple, limits: tuple, tau: float=_tau, cyclic: bool=_cyclic, threshold: float=_threshold)
```

## Description
Takes field like array and produces a vector centered at field peak position coordinates. Assumes at most one peak in the input field.

Note: make multiple peaks possible? Make cyclic possible? Implement expoential convergance to attractor

## Parameters-
- shape : tuple(Nx,Ny,...)
- limits : tuple((lx,ux), (ly,uy), ...)
- tau (optional) : float
    - If set, the output vector will exponentially converge to the peak position. If not, the vector jumps to the attractor in one time-step
    - Default = 0
- cyclic (optional) : bool
    - Default = False
- threshold (optional) : float
    - Default = 0.9

## Slots--
- in0 : jnp.array((Nx,Ny,...))
- out0 : jnp.array(len(Nx,Ny,...))

## Import

```python
from juniper import SpaceToRateCode
```
