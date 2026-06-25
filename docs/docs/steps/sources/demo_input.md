# DemoInput

```python
DemoInput(name: str, shape: tuple, sigma: tuple, amplitude: float, center=_center)
```

## Description
DemoInput is a GaussInput that can be customized during runtime.

## Parameters
- shape : tuple((Nx,Ny,...))
- sigma : float
- amplitude : amplitude

## Slots
- out0 : jnp.array(shape)

## Import

```python
from juniper import DemoInput
```
