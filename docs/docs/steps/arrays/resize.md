# Resize

```python
Resize(name: str, output_shape: tuple, interpolation: int = 1)
```

Resizes an array to `output_shape`.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` array |
| Outputs | `out0` resized array |

## Import

```python
from juniper import Resize
```

## Notes

- Uses JAX image resize behavior through the step implementation.
