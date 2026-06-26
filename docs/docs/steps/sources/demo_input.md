# DemoInput

```python
DemoInput(name: str, shape: tuple, sigma: tuple, amplitude: float, center=None)
```

Provides a generated demonstration input pattern.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | No input slots |
| Outputs | `out0` generated array with `shape` |

## Import

```python
from juniper import DemoInput
```

## Notes

- Use `GaussInput` or `CustomInput` for most explicit input definitions.
