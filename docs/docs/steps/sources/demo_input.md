# DemoInput

A Gaussian input source whose parameters (center, amplitude, sigma) can be modified at runtime. Useful for interactive demos where the stimulus position or strength needs to change during the simulation.

**Type:** Static (Source)

**Import:** `from juniper import DemoInput`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `shape` | `tuple` | Yes | Shape of the output field |
| `sigma` | `tuple` | Yes | Gaussian width per dimension (must match `shape` dimensionality) |
| `amplitude` | `float` | Yes | Gaussian peak amplitude |
| `center` | `tuple` | No | Center position. Default: `(0, 0, ...)` |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `out0` | Output | `shape` | Gaussian bump (recomputed each tick from current params) |

## Example

```python
di = DemoInput("di", {
    "shape": (50,),
    "sigma": (3,),
    "amplitude": 5,
    "center": (25,),
})
di >> field

# Change stimulus position at runtime
di._params["center"] = (30,)
```
