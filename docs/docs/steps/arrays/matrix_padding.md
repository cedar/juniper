# MatrixPadding

Pads a matrix with border elements in each dimension. Padding size and mode are configurable.

**Type:** Static

**Import:** `from juniper import MatrixPadding`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `border_size` | `int`, `tuple`, or nested `tuple` | Yes | Padding specification (see below) |
| `mode` | `str` | No | Padding mode as supported by `jax.numpy.pad`. Default: `"constant"` |

### Border Size Formats

- **`int`** -- Pad all dimensions equally on both sides.
- **`(before, after)`** -- Pad all dimensions with `before` elements before and `after` after.
- **`((b1, a1), (b2, a2), ...)`** -- Per-dimension before/after padding.

See `jax.numpy.pad` documentation for details.

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `(...)` | Padded array |

## Example

```python
pad = MatrixPadding("pad", {"border_size": 5})
# Pads 5 elements of zeros on each side of every dimension
source >> pad
```
