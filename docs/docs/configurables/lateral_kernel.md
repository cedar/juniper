# LateralKernel

```python
LateralKernel(params)
```

## Description
This class generates a lateral kernel object. Inputs should all be of the kernel type (include a get_kernel() method) and have the same dimensionality.
if the dimension sizes are not the same, the kernel with the largest dim size will be used and the others will be padded with zeros.

## Parameters-
- kernels : list([Gaussian])

## Import

```python
from juniper import LateralKernel
```
