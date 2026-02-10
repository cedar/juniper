# ScalarsToVector

Combines N separate scalar inputs into a single 1D vector of length N. Each scalar is received on a separate input slot (`in0`, `in1`, ..., `in{N-1}`).

**Type:** Static

**Import:** `from juniper import ScalarsToVector`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `N_scalars` | `int` | Yes | Number of scalar inputs (and length of output vector) |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` ... `in{N-1}` | Input | scalar | N separate scalar input slots |
| `out0` | Output | `(N_scalars,)` | Combined vector |

## Example

```python
s2v = ScalarsToVector("s2v", {"N_scalars": 3})
x_source >> s2v           # connects to in0
y_source >> "s2v.in1"     # connects to in1
z_source >> "s2v.in2"     # connects to in2
```
