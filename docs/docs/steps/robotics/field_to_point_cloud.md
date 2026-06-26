# FieldToPointCloud

```python
FieldToPointCloud(
    name: str,
    origin: tuple = (0.0, 0.0, 0.0),
    field_units_per_meter: tuple = (100.0, 100.0, 100.0),
    threshold: float = 0.9,
    N_pt=jnp.inf,
)
```

Converts active field cells into a fixed-size point cloud.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` field array |
| Outputs | `out0` point cloud with shape `(N_pt, 3)` |

## Import

```python
from juniper.robotics import FieldToPointCloud
```

## Notes

- Cells at or below `threshold` become zero-padding when fewer than `N_pt` points are active.
