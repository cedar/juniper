# PinHoleProjector

```python
PinHoleProjector(name: str, img_shape: tuple, focal_length: float, frustrum_angles: tuple)
```

Projects 3D points into pinhole-camera image coordinates.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` point cloud with shape `(N, 3)` |
| Outputs | `out0` projected coordinates |

## Import

```python
from juniper.robotics import PinHoleProjector
```
