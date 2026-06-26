# StaticDebug

```python
StaticDebug(name: str, shape: tuple = (1,))
```

A sink that stores its latest input for inspection and can force a branch to be evaluated.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` array |
| Outputs | `out0` pass-through/debug output |

## Import

```python
from juniper import StaticDebug
```

## Notes

- Primarily useful when checking static branches or runtime data movement.
