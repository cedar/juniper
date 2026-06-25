# TimedBoost

```python
TimedBoost(name: str, amplitude: float, duration: tuple)
```

## Description
Applies a homogenous boost to connected steps. Start and end of the boost can be specified.

## Parameters-
- amplitude : float
- duration [ms] : [start,stop]

## Slots-
- out0 : jnp.ndarray((1,))

## Import

```python
from juniper import TimedBoost
```
