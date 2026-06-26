# PinHoleBackProjector

```python
PinHoleBackProjector(name: str, img_shape: tuple, focal_length: float, frustrum_angles: tuple)
```

Back-projects image coordinates or range-image data into 3D points.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` image/range representation |
| Outputs | `out0` point cloud representation |

## Import

```python
from juniper.robotics import PinHoleBackProjector
```
