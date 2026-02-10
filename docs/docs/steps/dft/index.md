# Dynamic Field Theory Steps

These steps implement core Dynamic Field Theory (DFT) components -- neural fields with lateral interactions, Hebbian learning connections, and conversions between spatial and rate-coded representations.

All DFT steps are **dynamic** (they maintain internal state evolving over time).

| Step | Description |
|------|-------------|
| [NeuralField](neural_field.md) | Core neural field with sigmoid activation, lateral kernel, and noise |
| [HebbianConnection](hebbian_connection.md) | Learnable synaptic connection between fields with reward gating |
| [RateToSpaceCode](rate_to_space_code.md) | Converts a rate-coded vector to a Gaussian bump in field space |
| [SpaceToRateCode](space_to_rate_code.md) | Extracts peak position from a field as a rate-coded vector |
