# Gaussian

An N-dimensional Gaussian kernel object. Used as a kernel for neural field lateral interactions, convolution steps, and as the basis for `GaussInput` and `DemoInput` sources.

By default, the Gaussian is **factorized** (stored as a list of 1D factors per dimension) for efficient separable convolution. Set `factorized=False` to materialize the full N-D kernel.

**Import:** `from juniper import Gaussian`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sigma` | `tuple` | Yes | Standard deviation per dimension |
| `amplitude` | `float` | Yes | Peak amplitude |
| `normalized` | `bool` | Yes | If `True`, each 1D factor is normalized to sum to 1 |
| `shape` | `tuple` | No | Explicit kernel shape. If omitted, estimated from `sigma` (5 * sigma). |
| `max_shape` | `tuple` | No | Maximum allowed shape. The kernel is trimmed if it exceeds this. |
| `center` | `tuple` | No | Center of the Gaussian. Default: center of `shape`. |
| `factorized` | `bool` | No | Store as factorized 1D kernels for separable convolution. Default: `True` |

## Usage

### As a Lateral Kernel for NeuralField

```python
from juniper import Gaussian, NeuralField

nf = NeuralField("field", {
    "shape": (50,),
    # ...other params...
    "LateralKernel": Gaussian({
        "sigma": (3,),
        "amplitude": 5,
        "normalized": True,
        "max_shape": (50,),
    }),
})
```

### As a standalone kernel

```python
gauss = Gaussian({
    "sigma": (3, 3),
    "amplitude": 1,
    "normalized": False,
    "factorized": False,
})
kernel_array = gauss.get_kernel()  # Full 2D array
```

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_kernel()` | `jnp.ndarray` or `list` | The kernel tensor (full array if not factorized, list of 1D arrays if factorized) |
| `gen_convolve_func()` | `callable` | Returns a JIT-compiled convolution function using this kernel |
