# Resize

```python
Resize(name: str, output_shape: tuple, interpolation: int=_interpolation)
```

## Description
Resizes a Matrix to a new shape. Pixel values are interpolated using linear interpolation.

Note: Interpolation modes are currently limited to linear and nearest neighbor by jax. 
Note: Currently there is a bug where certain static input will initialize with shape [], leading to an error...

## Parameters
- output_shape : tuple(Nx,Ny,...)
- interpolation (optional) : int
    - Default = 0
    - 0 -> nearest neighbor
    - 1 -> linear

## Slots
- in0 : jnp.ndarray 
- out0 : jnp.ndarray

## Import

```python
from juniper import Resize
```
