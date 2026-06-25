# CoordinateTransformation

```python
CoordinateTransformation(name: str, FrameGraph, source_frame: str, target_frame: str)
```

## Description
Rigid coordinate transformation of incoming set of 3D vectors.

## Parameters
- FrameGraph : FrameGraph
- source_frame : str
- target_frame : str

## Slots
- in0 : jnp.array((N,3))
- in1 : jnp.array((3,))
- out0 : jnp.array((N,3))

## Import

```python
from juniper.robotics import CoordinateTransformation
```
