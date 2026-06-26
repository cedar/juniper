# NeuralField

```python
NeuralField(name: str, shape: tuple, sigmoid="AbsSigmoid", beta=100, theta=0, resting_level=-5, global_inhibition=0, input_noise_gain=0, tau=0.02, lateral_kernel=None)
```

Simulates a dynamic neural field with activation state, input drive, optional lateral interaction, global inhibition, resting level, and noise.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` field input |
| Outputs | `out0` transfer-function output; buffer `activation` stores the field state |

## Import

```python
from juniper import NeuralField
```

## Notes

- The activation update is integrated with the configured simulation time step.
- Use `lateral_kernel` for recurrent interaction.
