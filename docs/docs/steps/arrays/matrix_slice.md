# MatrixSlice

Extracts a sub-region of the input matrix using absolute index bounds for each dimension.

**Type:** Static

**Import:** `from juniper import MatrixSlice`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `slices` | `tuple((lower, upper), ...)` | Yes | Per-dimension slice bounds (absolute indices) |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `(...)` | Sliced sub-array |

## Example

```python
sl = MatrixSlice("sl", {"slices": ((10, 40), (5, 45))})
# From a (50,50) input, extracts rows 10-39 and columns 5-44
source >> sl
```
