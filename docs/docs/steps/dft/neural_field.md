# NeuralField

The core DFT building block. Implements a neural field with configurable sigmoid activation, lateral kernel interactions, global inhibition, and input noise. The field state evolves via Euler integration each tick.

The dynamics follow:

```
du/dt = (-u + h + lateral(sigmoid(u)) + g * sum(sigmoid(u)) + input) / tau + noise
```

Where `u` is the activation, `h` is the resting level, `g` is global inhibition, and `tau` is the time constant.

**Type:** Dynamic

**Import:** `from juniper import NeuralField`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `shape` | `tuple` | Yes | Dimensionality of the field, e.g., `(50,)` or `(50, 50)` |
| `sigmoid` | `str` | Yes | Activation function (see [Sigmoid](../../configurables/sigmoid.md)) |
| `beta` | `float` | Yes | Sigmoid steepness |
| `theta` | `float` | Yes | Sigmoid threshold |
| `resting_level` | `float` | Yes | Resting level `h` of the field (typically negative) |
| `global_inhibition` | `float` | Yes | Strength of global inhibition `g` (typically small and negative) |
| `input_noise_gain` | `float` | Yes | Amplitude of additive Gaussian noise |
| `tau` | `float` | Yes | Time constant in ms |
| `LateralKernel` | `LateralKernel` or `Gaussian` | No | Lateral interaction kernel. If omitted, no lateral interactions. |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `shape` | External input. Accepts unlimited connections (summed). Does not require any input connections. |
| `out0` | Output | `shape` | Sigmoided activation `sigmoid(u)` |

### Internal Buffers

| Buffer | Shape | Description |
|--------|-------|-------------|
| `activation` | `shape` | Raw activation `u` (recordable via `--recording field.activation`) |

## Example

```python
from juniper import NeuralField, Gaussian

nf = NeuralField("field", {
    "shape": (50,),
    "resting_level": -5,
    "global_inhibition": -0.01,
    "tau": 0.1,
    "input_noise_gain": 0.1,
    "sigmoid": "AbsSigmoid",
    "beta": 100,
    "theta": 0.5,
    "LateralKernel": Gaussian({
        "sigma": (3,),
        "amplitude": 5,
        "normalized": True,
        "max_shape": (50,),
    }),
})
input_step >> nf
```
