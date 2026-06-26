# Sigmoid

```python
Sigmoid({"sigmoid": name})
```

Selects a transfer function used by `NeuralField` and `TransferFunction`.

Supported names:

- `AbsSigmoid`
- `ExpSigmoid`
- `HeavySideSigmoid`
- `LinearSigmoid`
- `SemiLinearSigmoid`
- `LogarithmicSigmoid`

Most users pass the name directly to a step:

```python
field = jp.NeuralField("field", shape=(50,), sigmoid="ExpSigmoid", beta=100.0, theta=0.0)
```
