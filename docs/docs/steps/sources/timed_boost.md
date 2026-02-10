# TimedBoost

Outputs a scalar boost of a given amplitude during a specified time window. Outside the window, the output is zero. Useful for providing timed stimulation to neural fields.

**Type:** Dynamic (Source)

**Import:** `from juniper import TimedBoost`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `amplitude` | `float` | Yes | Boost value during the active window |
| `duration` | `list[start, stop]` | Yes | Time window in ms `[start, stop]`. Must have `start < stop`. |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `out0` | Output | `(1,)` | Scalar boost (amplitude during window, 0 otherwise) |

## Example

```python
boost = TimedBoost("boost", {
    "amplitude": 5.0,
    "duration": [100, 300],  # Active from 100ms to 300ms
})
boost >> field
```
