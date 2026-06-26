# MatrixSlice

```python
MatrixSlice(name: str, slices: tuple)
```

Slices an array using a tuple of Python slice specifications.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` array |
| Outputs | `out0` sliced array |

## Import

```python
from juniper import MatrixSlice
```

## Notes

- Use `slice(start, stop, step)` entries for normal Python slicing behavior.
