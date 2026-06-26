# PointCloudToField

```python
PointCloudToField(
    name: str,
    field_shape: tuple,
    origin: tuple = (0.0, 0.0, 0.0),
    field_units_per_meter: tuple = (100.0, 100.0, 100.0),
)
```

Rasterizes a point cloud into a field representation.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` point cloud with shape `(N, 3)` |
| Outputs | `out0` field array |

## Import

```python
from juniper.robotics import PointCloudToField
```
