# Projection

A combined expansion/compression and axis reordering step. Depending on whether the input has fewer, more, or equal dimensions compared to the output, it will expand, compress, or just reorder axes.

- **Expansion** (input dims < output dims): New axes are inserted at positions given by `axis`, sized according to `output_shape`, then axes are reordered by `order`.
- **Compression** (input dims > output dims): Axes specified by `axis` are reduced using `compression_type`, then remaining axes are reordered by `order`.
- **Reorder only** (input dims == output dims): Axes are permuted by `order`.

**Type:** Static

**Import:** `from juniper import Projection`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_shape` | `tuple` | Yes | Shape of the input |
| `output_shape` | `tuple` | Yes | Desired output shape |
| `axis` | `tuple` | Yes | Axes to expand into or compress from |
| `order` | `tuple` | Yes | Permutation order for the final reorder |
| `compression_type` | `str` | Yes | Aggregation for compression: `"Sum"`, `"Average"`, `"Maximum"`, `"Minimum"` |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `input_shape` | Input array |
| `out0` | Output | `output_shape` | Projected array |

## Example

```python
# Compress 3D (50,50,25) to 1D (50,) by summing axes 1 and 2
proj = Projection("proj", {
    "input_shape": (50, 50, 25),
    "output_shape": (50,),
    "axis": (1, 2),
    "order": (0,),
    "compression_type": "Sum",
})
source >> proj
```
