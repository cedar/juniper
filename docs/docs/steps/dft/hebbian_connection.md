# HebbianConnection

```python
HebbianConnection(name: str, source_shape: tuple[int, ...], target_shape: tuple[int, ...], tau: float=_tau, tau_decay: float=_tau_decay, learning_rate: float=_learning_rate, learning_rule: float=_learning_rule, bidirectional: bool=_bidirectional, reward_type: str=_reward_type, reward_duration: tuple[int, int]=_reward_duration, wheight_reset_slot: bool=_wheight_reset_slot)
```

## Description
Implements synaptic connections between the source and target field. Synaptic plasticity 
is implemented using either the instar or outstar hebbian learning rules, that may be gated 
by a reward signal. Length and delay of the reward signal can be customized. 

## Parameters
- source_shape : tuple((Nx,Ny,...))
- target_shape : tuple((Nx,...))
- tau (optional) : float
    - Default = 0.01
- tau_decay (optional) : float
    - Default = 0.1
- learning_rate (optional) : float
    - Default = 0.1
- learning_rule (optional) : str("instar", "outstar")
    - Default = instar
- bidirectional (optional) : bool
    - Default = True
- reward_type (optional) : str("no_reward", "reward_gated", "reward_interval")
    - Default = no_reward
- reward_duration (optional) : list[start,stop]
    - Default = [0,1]

## Slots
- in0: jnp.array(source_shape)
- in1: jnp.array(target_shape)
- in2: jnp.array((1,))
- in3 (optional): jnp.array((1,))
- out0: jnp.array(target_shape)
- out1: jnp.array(source_shape)

## Import

```python
from juniper import HebbianConnection
```
