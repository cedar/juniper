# ComponentMultiply

```python
ComponentMultiply(name: str)
```

Multiplies all incoming values on `in0` componentwise.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` accepts multiple arrays with compatible shapes |
| Outputs | `out0` product array |

## Import

```python
from juniper import ComponentMultiply
```

## Notes

- Unlike most steps, incoming values are aggregated by product instead of sum.
