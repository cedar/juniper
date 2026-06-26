# AddConstant

```python
AddConstant(name: str, constant: float)
```

Adds `constant` elementwise to `in0`.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` array |
| Outputs | `out0` array with the same shape as `in0` |

## Import

```python
from juniper import AddConstant
```

## Notes

- Useful for shifting activation levels or adding a bias.
