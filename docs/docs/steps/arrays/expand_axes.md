# ExpandAxes

Expands the input array by inserting new dimensions at specified positions and repeating values along them.

**Type:** Static

**Import:** `from juniper import ExpandAxes`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `axis` | `tuple` | Yes | Positions where new dimensions are inserted |
| `sizes` | `tuple` | Yes | Size of each new dimension (same order as `axis`) |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `(...)` | Expanded array with new dimensions |

## Example

```python
expand = ExpandAxes("expand", {"axis": (1,), "sizes": (10,)})
# Input shape (50,) -> Output shape (50, 10)
source >> expand
```
