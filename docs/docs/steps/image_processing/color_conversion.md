# ColorConversion

```python
ColorConversion(name: str, channels: str = "RGB")
```

Splits or converts color channels according to the configured channel mode.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` image-like array |
| Outputs | `out0`, `out1`, `out2` channel outputs |

## Import

```python
from juniper import ColorConversion
```

## Notes

- Use `RGB2HSV` when a direct RGB-to-HSV conversion is needed.
