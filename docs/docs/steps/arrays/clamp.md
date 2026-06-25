# Clamp

```python
Clamp(name: str, limits: tuple)
```

## Description
Clamps the values in an array into the range specified by min and max limits.

Note: Add ability to replace clipped elements with custom values.

## Parameters
- limits: tuple(min,max)

## Slots
- in0 : jnp.array()
- out0 : jnp.array()

## Import

```python
from juniper import Clamp
```
