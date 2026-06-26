# CoordinateTransformation

```python
CoordinateTransformation(name: str, FrameGraph, source_frame: str, target_frame: str)
```

Transforms a point cloud from one named frame to another using a `FrameGraph`.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` points with shape `(N, 3)`; `in1` joint state |
| Outputs | `out0` transformed points |

## Import

```python
from juniper.robotics import CoordinateTransformation
```
