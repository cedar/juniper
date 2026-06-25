# NeuralField

```python
NeuralField(name: str, shape: tuple[int, ...], sigmoid: str=_sigmoid, beta: int=_beta, theta: float=_theta, resting_level: float=_resting_level, global_inhibition: float=_global_inhibition, input_noise_gain: float=0, tau: float=_tau, lateral_kernel: LateralKernel | None=_lateral_kernel)
```

## Description
Neural Field step. 

## Parameters    
- shape : tuple(Nx,Ny,...)
- sigmoid (optional) : str(AbsSigmoid, HeavySideSigmoid, ExpSigmoid, LinearSigmoid, SemiLinearSigmoid, LogarithmicSigmoid)
- beta (optional) : float
- theta (optional) : float
- resting_level (optional) : float
- global_inhibition (optional) : float
- input_noise_gain (optional) : float
- tau (optional) [ms] : float
- LateralKernel (optional) : LateralKernel or Gaussian

## Slots
- Input: jnp.ndarray(shape)
- output: jnp.ndarray(shape)

## Import

```python
from juniper import NeuralField
```

## Example

```python
field = NeuralField("field", shape=(50,), lateral_kernel=Gaussian({"shape": (50,), "sigma": (3,), "amplitude": 5.0}))
```
