# VectorToScalars

Splits a 1D input vector of length N into N separate scalar outputs. Each scalar is written to a separate output slot (`out0`, `out1`, ..., `out{N-1}`).

**Type:** Static

**Import:** `from juniper import VectorToScalars`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `N_scalars` | `int` | Yes | Number of scalars (must match input vector length) |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(N_scalars,)` | Input vector |
| `out0` ... `out{N-1}` | Output | scalar | N separate scalar output slots |

## Example

```python
v2s = VectorToScalars("v2s", {"N_scalars": 3})
vec_source >> v2s
v2s.o0 >> x_consumer  # first element
v2s.o1 >> y_consumer  # second element
v2s.o2 >> z_consumer  # third element
```
