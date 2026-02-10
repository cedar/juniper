# RateToSpaceCode

Converts a rate-coded vector (representing a position in metric space) into a Gaussian bump centered at the corresponding field coordinates. Useful for injecting position information into a neural field.

**Type:** Static

**Import:** `from juniper import RateToSpaceCode`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `shape` | `tuple` | Yes | Shape of the output field |
| `limits` | `tuple((low, high), ...)` | Yes | Metric range per dimension |
| `center` | `tuple` | No | Center offset. Default: midpoint of limits |
| `amplitude` | `float` | No | Gaussian peak amplitude. Default: `1.0` |
| `sigma` | `tuple` | No | Gaussian width per dimension. Default: `(1.0, ...)` |
| `cyclic` | `bool` | No | Cyclic mode (not yet implemented). Default: `False` |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(len(shape),)` | Rate-coded position vector |
| `out0` | Output | `shape` | Gaussian bump field |

## Example

```python
r2s = RateToSpaceCode("r2s", {
    "shape": (50, 50),
    "limits": ((0, 50), (0, 50)),
    "amplitude": 2,
    "sigma": (3, 3),
})
position_vector >> r2s >> neural_field
```
