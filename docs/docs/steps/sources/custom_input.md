# CustomInput

A source step whose output can be set programmatically from external code by modifying the `output` attribute. Useful for injecting data from Jupyter notebooks or custom control loops.

**Type:** Static (Source)

**Import:** `from juniper import CustomInput`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `shape` | `tuple` | Yes | Shape of the output array |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `out0` | Output | `shape` | Output array (set via `step.output`) |

## Example

```python
import jax.numpy as jnp

ci = CustomInput("ci", {"shape": (50,)})
ci >> some_step

# Later, set the output
ci.output = jnp.ones((50,)) * 3.0
```
