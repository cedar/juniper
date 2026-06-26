# BCMConnection

```python
BCMConnection(name: str, source_shape: tuple, target_shape: tuple, tau_weights=1.0, tau_theta=1.0, learning_rate=0.1, min_theta=0.0, use_fixed_theta=True, fixed_theta=0.25, norm_target=0.0, norm_rate=0.0, safeguard_thr=-1.0, theta_eps=1e-6)
```

Implements a BCM-style learned connection between source and target activations.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` source activation; `in1` target activation; `in2` reward or modulation signal |
| Outputs | `out0` target-shaped output |

## Import

```python
from juniper import BCMConnection
```

## Notes

- Permanent buffers `wheights` and `theta` store learned weights and thresholds.
- The first two dimensions of `source_shape` and `target_shape` must match.
