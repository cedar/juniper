# MatrixPadding

```python
MatrixPadding(name: str, border_size, mode: str=_mode)
```

## Description
Padds a Matrix by a number of elements in each dimension.

Note: Add ability to pad to specified size.

## Parameters
- border_size : [int | Array | jnp.ndarray])
    - Size of border for each dimension.
    - int or (int,): pad each array dimension with the same number of values both before and after.
    - (before, after): pad each array with before elements before, and after elements after.
    - ((before_1, after_1), (before_2, after_2), ... (before_N, after_N)): specify distinct before and after values for each array dimension.
    - See jax.numpy.pad documentation for reference
- mode (optional) : str
    - Specifies by what mode the padded values are chosen.
    - See available modes in jax.numpy.pad documentation.
    - Default = "constant"

## Slots
- Input: jnp.ndarray 
- output: jnp.ndarray

## Import

```python
from juniper import MatrixPadding
```
