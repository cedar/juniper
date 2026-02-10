# CompressAxes

Reduces (compresses) the input array along specified axes using an aggregation function.

**Type:** Static

**Import:** `from juniper import CompressAxes`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `axis` | `tuple` | Yes | Axes to reduce, e.g. `(2,)` |
| `compression_type` | `str` | Yes | Aggregation function (see below) |

### Supported Compression Types

| Value | Operation |
|-------|-----------|
| `"Sum"` | Sum along axis |
| `"Average"` | Mean along axis |
| `"Maximum"` | Max along axis |
| `"Minimum"` | Min along axis |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `(...)` | Array with specified axes removed |

## Example

```python
comp = CompressAxes("comp", {"axis": (2,), "compression_type": "Maximum"})
# Input shape (50,50,25) -> Output shape (50,50)
source >> comp
```
