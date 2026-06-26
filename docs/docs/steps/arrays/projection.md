# Projection

```python
Projection(
    name: str,
    input_shape: tuple,
    output_shape: tuple,
    axis: tuple,
    order: tuple,
    compression_type: str,
)
```

Projects an input array into a different dimensionality by reducing or expanding axes, then reordering axes.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` array with `input_shape` |
| Outputs | `out0` array with `output_shape` |

## Import

```python
from juniper import Projection
```

## Notes

- `compression_type` supports `Sum`, `Average`, `Maximum`, and `Minimum`.
- In a contraction, `axis` selects the input axes to reduce.
- In an expansion, `axis` selects the positions where new axes are inserted before `order` is applied.
- For simple reductions, prefer `CompressAxes`.
