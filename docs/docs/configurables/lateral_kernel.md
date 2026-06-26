# LateralKernel

```python
LateralKernel({"kernels": [kernel_a, kernel_b, ...]})
```

Combines several kernel objects into one lateral interaction kernel. This is commonly used to build local excitation and broader inhibition for a `NeuralField`.

All input kernels must have the same dimensionality and must use the same representation: either all factorized or all full kernels. Kernels with smaller shapes are padded to match the largest shape before combination.

## Example

```python
import juniper as jp

excitation = jp.Gaussian({"sigma": (3,), "amplitude": 5.0, "normalized": True})
inhibition = jp.Gaussian({"sigma": (10,), "amplitude": -2.0, "normalized": True})
kernel = jp.LateralKernel({"kernels": [excitation, inhibition]})

field = jp.NeuralField("field", shape=(100,), lateral_kernel=kernel)
```
