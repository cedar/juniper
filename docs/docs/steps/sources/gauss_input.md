# GaussInput

Generates a static Gaussian bump as input. The kernel is computed once at initialization and does not change during simulation.

**Type:** Static (Source)

**Import:** `from juniper import GaussInput`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `shape` | `tuple` | Yes | Shape of the output field |
| `sigma` | `tuple` | Yes | Gaussian width per dimension (must match `shape` dimensionality) |
| `amplitude` | `float` | Yes | Gaussian peak amplitude |
| `center` | `tuple` | No | Center position. Default: center of `shape` |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `out0` | Output | `shape` | Gaussian bump (constant) |

## Example

```python
gi = GaussInput("gi", {
    "shape": (50,),
    "sigma": (3,),
    "amplitude": 5,
    "center": (25,),
})
gi >> field
```
