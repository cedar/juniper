# CompressAxes

```python
CompressAxes(name: str, axis: tuple, compression_type: str, compress_all: bool=_compress_all)
```

## Description
Compress incoming step along specified dimension.

## Parameters
- axis : tuple(ax0,ax1,...)
- compression_type : str(Sum,Average,Maximum,Minimum)
- compress_all (optional) : bool
    - flag to indicate that all input axes will be supressed. This is needed to establish valid output shape.

## Slots
- in0 : jnp.array()
- out0 : jnp.array()

## Import

```python
from juniper import CompressAxes
```
