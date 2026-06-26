# Convolution

```python
Convolution(name: str, kernel=None, mode: str = "same")
```

Convolves `in0` with a static or dynamic kernel using JAX FFT convolution.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` input array; `in1` kernel array when `kernel=None` |
| Outputs | `out0` convolved array |

## Import

```python
from juniper import Convolution
```

## Notes

- `kernel` can be a configurable object such as `Gaussian` or `LateralKernel`.
- `mode` follows `jax.scipy.signal.fftconvolve`: `same`, `full`, or `valid`.
