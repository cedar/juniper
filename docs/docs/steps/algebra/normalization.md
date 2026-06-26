# Normalization

```python
Normalization(name: str, function: str)
```

Normalizes `in0` by its norm.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` array |
| Outputs | `out0` normalized array |

## Import

```python
from juniper import Normalization
```

## Notes

- Supported functions are `InfinityNorm`, `L1Norm`, and `L2Norm`.
