# Sigmoid

A wrapper that maps a sigmoid name to its implementation function. Used internally by `NeuralField` and `TransferFunction`. Not typically instantiated directly.

**Import:** `from juniper.configurables.Sigmoid import Sigmoid`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sigmoid` | `str` | Yes | Name of the sigmoid function |

## Supported Sigmoid Functions

| Name | Description |
|------|-------------|
| `AbsSigmoid` | Absolute value sigmoid |
| `HeavySideSigmoid` | Heaviside step function |
| `ExpSigmoid` | Exponential sigmoid |
| `LinearSigmoid` | Linear sigmoid |
| `SemiLinearSigmoid` | Semi-linear (ramp) sigmoid |
| `LogarithmicSigmoid` | Logarithmic sigmoid |

All sigmoid functions take the signature `sigmoid(x, beta, theta)` where `beta` controls steepness and `theta` is the threshold.

## Usage

Sigmoids are typically selected via a string parameter in steps:

```python
# In NeuralField
nf = NeuralField("field", {
    "sigmoid": "AbsSigmoid",
    "beta": 100,
    "theta": 0.5,
    # ...
})

# In TransferFunction
tf = TransferFunction("tf", {
    "function": "ExpSigmoid",
    "beta": 1.0,
    "threshold": 0.0,
})
```
