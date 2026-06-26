# ColorFMap

```python
ColorFMap(name: str, bins: int, saturation_threshold=0.0, hue_range=360, value_threshold=0.0)
```

Maps color information to feature-map activations.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` hue-like input; `in1` saturation-like input; `in2` value-like input |
| Outputs | `out0` feature map |

## Import

```python
from juniper import ColorFMap
```

## Notes

- Pixels below saturation or value thresholds are suppressed.
