# Flip

Reverses the order of elements along specified axes.

**Type:** Static

**Import:** `from juniper import Flip`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `axis` | `tuple` | Yes | Axes along which to flip |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `(...)` | Flipped array (same shape) |

## Example

```python
flip = Flip("flip", {"axis": (0, 1)})
source >> flip
```
