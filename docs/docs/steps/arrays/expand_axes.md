# ExpandAxes

```python
ExpandAxes(name: str, axis: tuple, sizes: tuple)
```

Inserts axes and repeats values along those axes.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` array |
| Outputs | `out0` expanded array |

## Import

```python
from juniper import ExpandAxes
```

## Notes

- Each entry in `axis` pairs with the corresponding entry in `sizes`.
