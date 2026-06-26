# Dynamic Field Theory

Dynamic Field Theory models activation as a continuous state over a metric space. A neural field receives input, relaxes toward a resting level, interacts laterally with nearby positions, and produces an output through a transfer function.

For a field activation `u`, a common form is:

```text
tau * du/dt = -u + h + input + lateral_interaction + global_inhibition + noise
output = sigmoid(u)
```

JUNIPER implements this as fixed-step simulation with JAX arrays. 

| Step | Purpose |
|------|---------|
| [`NeuralField`](neural_field.md) | Dynamic activation field. |
| [`SpaceToRateCode`](space_to_rate_code.md) | Convert a spatial peak to a compact value. |
| [`RateToSpaceCode`](rate_to_space_code.md) | Convert a compact value to a spatial activation pattern. |
| [`HebbianConnection`](hebbian_connection.md) | Learned associative connection. |
| [`BCMConnection`](bcm_connection.md) | BCM-style learned connection. |
