# RateToSpaceCode

```python
RateToSpaceCode(
    name: str,
    shape: tuple,
    limits: tuple,
    center: tuple | None = None,
    amplitude: float = 1.0,
    sigma: tuple | None = None,
    cyclic: bool = False,
)
```

Converts a scalar or low-dimensional rate code into a spatial Gaussian activation pattern.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` rate-coded value |
| Outputs | `out0` spatial activation with `shape` |

## Import

```python
from juniper import RateToSpaceCode
```

## Notes

- `limits` define the represented value range.
- If `center` is omitted, the midpoint of each limit interval is used.
- If `sigma` is omitted, a width of `1.0` is used in each dimension.
