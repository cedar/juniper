# CompressAxes

```python
CompressAxes(name: str, axis: tuple, compression_type: str, compress_all: bool = False)
```

Reduces selected axes using a reduction function.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` array |
| Outputs | `out0` reduced array |

## Import

```python
from juniper import CompressAxes
```

## Notes

- Supported reductions are `Sum`, `Average`, `Maximum`, and `Minimum`.
- Set `compress_all=True` when all input axes are reduced and a length-one output is required.
