# BCMConnection

```python
BCMConnection(name: str, source_shape: tuple, target_shape: tuple, tau_weights: float=_tau_weights, tau_theta: float=_tau_theta, learning_rate: float=_learning_rate, min_theta: float=_min_theta, use_fixed_theta: bool=_use_fixed_theta, fixed_theta: float=_fixed_theta, norm_target: float=_norm_target, norm_rate: float=_norm_rate, safeguard_thr: float=_safeguard_thr, theta_eps: float=_theta_eps)
```

## Description
Implements a BCM-style synaptic connection between source and target fields.
Weights are learned from source and target activity when the reward input is
active. The target trace theta is either updated dynamically or clamped to a
fixed value, depending on the use_fixed_theta setting.

## Parameters
- source_shape : tuple((Nx,Ny,Nf))
    - Source field shape.
- target_shape : tuple((Nx,Ny,Nd))
    - Target field shape. The first two dimensions must match shape.
- tau_weights (optional) : float
    - Default = 1.0
- tau_theta (optional) : float
    - Default = 1.0
- learning_rate (optional) : float
    - Default = 0.1
- min_theta (optional) : float
    - Default = 0.0
- use_fixed_theta (optional) : bool
    - Default = True
- fixed_theta (optional) : float
    - Default = 0.25
- norm_target (optional) : float
    - Default = 0.0
- norm_rate (optional) : float
    - Default = 0.0
- safeguard_thr (optional) : float
    - Default = -1.0
- theta_eps (optional) : float
    - Default = 1e-6

## Slots
- in0: jnp.array(shape)
- in1: jnp.array(target_shape)
- in2: jnp.array((1,))
- out0: jnp.array(target_shape)

## Import

```python
from juniper import BCMConnection
```
