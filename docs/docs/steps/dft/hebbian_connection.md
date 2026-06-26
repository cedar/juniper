# HebbianConnection

```python
HebbianConnection(
    name: str,
    source_shape: tuple,
    target_shape: tuple,
    tau: float = 10 * delta_t,
    tau_decay: float = 100 * delta_t,
    learning_rate: float = 0.1,
    learning_rule: str = "instar",
    bidirectional: bool = True,
    reward_type: str = "no_reward",
    reward_duration: tuple = (0, 1),
    wheight_reset_slot: bool = False,
)
```

Connects source and target activations with a learned weight tensor.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` source activation; `in1` target activation; `in2` reward signal; optional `in3` reset signal |
| Outputs | `out0` target-directed output; `out1` reverse output |

## Import

```python
from juniper import HebbianConnection
```

## Notes

- Permanent buffer `wheights` stores learned weights.
- Buffers `reward_timer` and `reward_onset` track reward timing.
