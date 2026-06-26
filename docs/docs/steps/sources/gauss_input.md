# GaussInput

```python
GaussInput(name: str, shape: tuple, sigma: tuple, amplitude: float, center=None)
```

Provides a static Gaussian activation pattern.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | No input slots |
| Outputs | `out0` Gaussian array with `shape` |

## Import

```python
from juniper import GaussInput
```

## Notes

- If `center` is omitted, the Gaussian is centered in the output shape.
