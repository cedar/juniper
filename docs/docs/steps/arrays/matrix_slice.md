# MatrixSlice

```python
MatrixSlice(name: str, slices: tuple)
```

## Description
Slices Matrix according to specified slice ranges.

Note: Add ability to choose center cutout as a slice mode

## Parameters
- slices: tuple((lower,upper), ...)
    - For each dimension slices specifies the lower and upper indice bounds for slicing. 
    - Absolute indice coordinates are used. So (0,10) will slice the first 10 elements (not 10 in the center).

## Slots
- Input: jnp.ndarray 
- output: jnp.ndarray

## Import

```python
from juniper import MatrixSlice
```
