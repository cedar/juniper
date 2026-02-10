# TransferFunction

Applies a sigmoid non-linearity to the input. The sigmoid type, steepness (beta), and threshold are configurable.

**Type:** Static

**Import:** `from juniper import TransferFunction`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `function` | `str` | Yes | Sigmoid type (see below) |
| `beta` | `float` | Yes | Steepness of the sigmoid |
| `threshold` | `float` | Yes | Threshold / inflection point |

### Supported Sigmoid Functions

`AbsSigmoid`, `HeavySideSigmoid`, `ExpSigmoid`, `LinearSigmoid`, `SemiLinearSigmoid`, `LogarithmicSigmoid`

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `(...)` | sigmoid(input, beta, threshold) |

## Example

```python
tf = TransferFunction("tf", {
    "function": "ExpSigmoid",
    "threshold": 0.0,
    "beta": 1.0,
})
source >> tf
```
