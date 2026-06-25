# Sum

```python
Sum(name: str)
```

## Description
Adds incoming steps component wise.

## Parameters-

## Slots-
- in0 : jnp.ndarray 
- out0 : jnp.ndarray

## Import

```python
from juniper import Sum
```

## Example

```python
a >> total
b >> total
c >> total  # inputs are summed on total.in0
```
