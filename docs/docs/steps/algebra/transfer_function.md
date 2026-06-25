# TransferFunction

```python
TransferFunction(name: str, threshold: float, beta: float, function: str)
```

## Description
Applies a non-linearity.

## Parameters-
- threshold : float
- beta : float
- function : str(AbsSigmoid, HeavySideSigmoid, ExpSigmoid, LinearSigmoid, SemiLinearSigmoid, LogarithmicSigmoid)

## Slots-
- in0 : jnp.ndarray 
- out0 : jnp.ndarray

## Import

```python
from juniper import TransferFunction
```
