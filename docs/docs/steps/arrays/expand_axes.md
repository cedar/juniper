# ExpandAxes

```python
ExpandAxes(name: str, axis: tuple, sizes: tuple)
```

## Description
Expand incoming step along specified axis.

## Parameters
- axis : tuple(ax0,ax1,...)
- sizes : tuple(s0,s1,...)
    - sizes per dimension

## Slots
- in0 : jnp.array((Nx,...))
- out0 : jnp.array((Nx,ax0,ax1,...))

## Import

```python
from juniper import ExpandAxes
```
