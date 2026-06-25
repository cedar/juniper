# ScalarsToVector

```python
ScalarsToVector(name: str, N_scalars: int)
```

## Description
Turns a number of scalars into a 1d-Array (vector).

Note: Make it possible to have incomplete incoming connections.

## Parameters-
- N_scalars: int 
    - Number of scalars (length of output Vector)

## Slots-
- [in0, in1, ..., in{N_scalars-1}] : jnp.ndarray 
    - N_scalars separate inputs indexed by 'in{i}'
- out0 : jnp.ndarray 
    - Vector of length N_scalars

## Import

```python
from juniper import ScalarsToVector
```
