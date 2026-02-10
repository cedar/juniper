# Clamp

Clamps all values in the input array to lie within a specified range `[min, max]`.

**Type:** Static

**Import:** `from juniper import Clamp`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limits` | `tuple(min, max)` | Yes | Lower and upper bounds for clamping |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `(...)` | Clamped array |

## Example

```python
clamp = Clamp("clamp", {"limits": (0.0, 1.0)})
source >> clamp  # All values outside [0, 1] are clipped
```
