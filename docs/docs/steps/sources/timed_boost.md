# TimedBoost

```python
TimedBoost(name: str, amplitude: float, duration: tuple)
```

Outputs a scalar boost during a configured time interval.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | No input slots |
| Outputs | `out0` scalar-like boost signal |

## Import

```python
from juniper import TimedBoost
```

## Notes

- Uses buffer `local_time` to track elapsed simulation time.
