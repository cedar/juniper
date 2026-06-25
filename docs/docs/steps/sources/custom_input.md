# CustomInput

```python
CustomInput(name: str, shape: tuple)
```

## Description
Custom Input, can be set from outside by modifying self.output.

## Parameters
- shape : tuple((Nx,Ny,...))

## Slots
- out0 : jnp.array(shape)

## Import

```python
from juniper import CustomInput
```

## Example

```python
source = CustomInput("source", shape=(2,))
source.set_data([1.0, 2.0])
```
