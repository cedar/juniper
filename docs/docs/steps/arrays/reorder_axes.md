# ReorderAxes

Permutes (transposes) the axes of the input array.

**Type:** Static

**Import:** `from juniper import ReorderAxes`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `order` | `tuple` | Yes | New axis order, e.g., `(1, 0)` to swap two axes |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `(...)` | Transposed array |

## Example

```python
reorder = ReorderAxes("reorder", {"order": (1, 0)})
# Transposes a 2D matrix
source >> reorder
```
