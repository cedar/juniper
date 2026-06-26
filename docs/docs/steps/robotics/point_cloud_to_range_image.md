# PointCloudToRangeImage

```python
PointCloudToRangeImage(name: str, image_shape: tuple, pan_range: tuple, tilt_range: tuple)
```

Projects a point cloud into a range image.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` point cloud with shape `(N, 3)` |
| Outputs | `out0` range image with `image_shape` |

## Import

```python
from juniper.robotics import PointCloudToRangeImage
```
