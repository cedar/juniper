# Normalization

Normalizes the input array using a specified norm function. A small epsilon (1e-8) is added to avoid division by zero.

**Type:** Static

**Import:** `from juniper import Normalization`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `function` | `str` | Yes | Norm type. One of: `"InfinityNorm"`, `"L1Norm"`, `"L2Norm"` |

### Supported Norms

| Value | Norm |
|-------|------|
| `"InfinityNorm"` | Maximum absolute value |
| `"L1Norm"` | Sum of absolute values |
| `"L2Norm"` | Euclidean norm |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `(...)` | Normalized array |

## Example

```python
norm = Normalization("norm", {"function": "L2Norm"})
source >> norm
```
