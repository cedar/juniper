# StaticGain

Multiplies every element of the input by a constant scalar factor.

**Type:** Static

**Import:** `from juniper import StaticGain`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `factor` | `float` | Yes | Scalar multiplier |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `(...)` | Input * factor |

## Example

```python
gain = StaticGain("gain", {"factor": 0.5})
source >> gain
```
