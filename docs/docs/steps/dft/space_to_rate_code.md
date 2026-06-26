# SpaceToRateCode

```python
SpaceToRateCode(
    name: str,
    shape: tuple,
    limits: tuple,
    tau: float = 0,
    cyclic: bool = False,
    threshold: float = 0.9,
)
```

Converts a spatial activation pattern into a scalar rate code, typically by estimating a peak position within `limits`.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` spatial activation |
| Outputs | `out0` rate-coded value; buffer `peak_pos` stores the peak estimate |

## Import

```python
from juniper import SpaceToRateCode
```

## Notes

- Useful when a field represents a continuous variable and another component needs a compact value.
