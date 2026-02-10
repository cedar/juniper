# LateralKernel

Combines multiple kernel objects (typically `Gaussian` instances) into a single lateral interaction kernel. Used as the `LateralKernel` parameter for `NeuralField`. All input kernels must have the same dimensionality and the same factorization setting.

If kernels have different sizes, smaller kernels are zero-padded to match the largest.

**Import:** `from juniper import LateralKernel`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `kernels` | `list[Gaussian]` | Yes | List of kernel objects to combine |

## Usage

A typical DFT lateral kernel has a Mexican-hat profile: a narrow excitatory Gaussian and a broad inhibitory Gaussian.

```python
from juniper import Gaussian, LateralKernel, NeuralField

excitatory = Gaussian({
    "sigma": (3,), "amplitude": 5, "normalized": True, "max_shape": (50,)
})
inhibitory = Gaussian({
    "sigma": (8,), "amplitude": -2, "normalized": True, "max_shape": (50,)
})

lk = LateralKernel({"kernels": [excitatory, inhibitory]})

nf = NeuralField("field", {
    "shape": (50,),
    # ...other params...
    "LateralKernel": lk,
})
```

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_kernel()` | `jnp.ndarray` or `list` | The combined kernel |
| `gen_convolve_func()` | `callable` | Returns a JIT-compiled convolution function |
