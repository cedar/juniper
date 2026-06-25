# VectorToScalars

```python
VectorToScalars(name: str, N_scalars: int)
```

## Description
Turns a 1d-Array (Vector) into a set of individual scalars.

## Parameters
- N_scalars: int 
    - Number of scalars (length of input Vector)

## Slots
- in0: jnp.ndarray(N_scalars)
    - 1d-Array of length N_scalars
- [out0, out1, ..., out{N_scalars-1}]: jnp.ndarray((1,))
    - separate outputs indexed by 'out{i}'

## Import

```python
from juniper import VectorToScalars
```
