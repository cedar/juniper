# CustomInput

```python
CustomInput(name: str, shape: tuple)
```

Provides Python-side data to the simulation.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | No input slots |
| Outputs | `out0` array with `shape` |

## Import

```python
from juniper import CustomInput
```

## Notes

- Set data with `set_data(array)`. Data is pushed before each tick, so values can be changed between simulation runs.
