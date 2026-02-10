# HebbianConnection

Implements learnable synaptic connections between a source and target field. Supports instar and outstar Hebbian learning rules, optional bidirectional output, and reward gating with configurable timing.

The weight matrix evolves over time using the selected learning rule, modulated by a reward signal.

**Type:** Dynamic

**Import:** `from juniper import HebbianConnection`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `shape` | `tuple` | Yes | Shape of the source field |
| `target_shape` | `tuple` | Yes | Shape of the target field |
| `tau` | `float` | No | Buildup time constant. Default: `0.01` |
| `tau_decay` | `float` | No | Decay time constant. Default: `0.1` |
| `learning_rate` | `float` | No | Learning rate. Default: `0.1` |
| `learning_rule` | `str` | No | `"instar"` or `"outstar"`. Default: `"instar"` |
| `bidirectional` | `bool` | No | Enable reverse output. Default: `True` |
| `reward_type` | `str` | No | Reward gating mode (see below). Default: `"no_reward"` |
| `reward_duration` | `list[start, stop]` | No | Reward timing window. Default: `[0, 1]` |

### Reward Types

| Value | Description |
|-------|-------------|
| `"no_reward"` | Learning is always active |
| `"reward_gated"` | Learning only when reward signal > 0.9 |
| `"reward_interval"` | Learning during a timed window after reward onset |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `shape` | Source field activation |
| `in1` | Input | `target_shape` | Target field activation |
| `in2` | Input | `(1,)` | Reward signal |
| `out0` | Output | `target_shape` | Forward output (weight * source) |
| `out1` | Output | `shape` | Reverse output (bidirectional, weight * target) |

### Internal Buffers

| Buffer | Shape | Description |
|--------|-------|-------------|
| `wheights` | `shape + target_shape` | Weight matrix (saved across runs) |
| `reward_timer` | `(1,)` | Internal reward timing state |
| `reward_onset` | `(1,)` | Internal reward onset flag |

## Example

```python
hebb = HebbianConnection("hebb", {
    "shape": (50,),
    "target_shape": (50,),
    "learning_rule": "instar",
    "reward_type": "no_reward",
    "tau": 0.01,
    "tau_decay": 0.1,
    "learning_rate": 0.1,
    "bidirectional": True,
    "reward_duration": [0, 1],
})

source_field >> hebb                  # in0: source
target_field >> "hebb.in1"            # in1: target
reward_signal >> "hebb.in2"           # in2: reward
hebb >> some_consumer                 # out0: forward
hebb.o1 >> reverse_consumer           # out1: reverse
```
