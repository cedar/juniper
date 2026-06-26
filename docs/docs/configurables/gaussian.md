# Gaussian

```python
Gaussian(params)
```

Creates an n-dimensional Gaussian kernel. Gaussian kernels are used by `GaussInput`, `Convolution`, and `NeuralField`.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `sigma` | Tuple of standard deviations, one per dimension. |
| `amplitude` | Kernel amplitude. Negative amplitudes create inhibitory kernels. |
| `normalized` | If `True`, normalize each component before applying amplitude. |
| `shape` | Optional explicit kernel shape. If omitted, a size is estimated from `sigma`. |
| `center` | Optional center index. Defaults to the center of `shape`. |
| `max_shape` | Optional maximum shape. Oversized kernels are cropped. |
| `factorized` | If `True`, store separable one-dimensional factors. Defaults to `True`. |

## Example

```python
import juniper as jp

kernel = jp.Gaussian({
    "shape": (50,),
    "sigma": (3,),
    "amplitude": 5.0,
    "normalized": True,
    "factorized": True,
})
```

Use `factorized=False` when a full materialized kernel is required.
