# RemoveBlackWhiteGreys

```python
RemoveBlackWhiteGreys(
    name: str,
    saturation_threshold: float = 0.2,
    value_threshold: float = 0.2,
)
```

Suppresses black, white, and grey pixels based on saturation and value thresholds.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` image or HSV-like data |
| Outputs | `out0` filtered image data |

## Import

```python
from juniper import RemoveBlackWhiteGreys
```
