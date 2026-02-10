# Resize

Resizes the input array to a new shape using interpolation. Supports nearest-neighbor and linear interpolation.

**Type:** Static

**Import:** `from juniper import Resize`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `output_shape` | `tuple` | Yes | Target shape |
| `interpolation` | `int` | No | `0` = nearest neighbor (default), `1` = linear |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `output_shape` | Resized array |

## Example

```python
resize = Resize("resize", {"output_shape": (100, 100), "interpolation": 1})
source >> resize
```
