# ComponentMultiply

```python
ComponentMultiply(name: str)
```

## Description
Componentwise multiplication of incoming steps.

## Parameters

## Slots
- in0 : jnp.array()
- out0 : jnp.array()

## Import

```python
from juniper import ComponentMultiply
```

## Example

```python
a >> product
b >> product  # inputs are multiplied componentwise
```
