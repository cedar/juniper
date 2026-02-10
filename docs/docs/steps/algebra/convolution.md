# Convolution

Convolves the input array with a kernel. The kernel can be a static object (e.g., a `Gaussian` or `LateralKernel`) passed as a parameter, or a dynamic kernel received via the `kernel` input slot.

**Type:** Static

**Import:** `from juniper import Convolution`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `kernel` | `LateralKernel` or `Gaussian` | No | Static kernel object. If omitted, a dynamic kernel is expected via the `kernel` input slot. |
| `mode` | `str` | No | Convolution mode. Default: `"same"`. |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array to convolve |
| `kernel` | Input | `(...)` | Dynamic kernel (only used if no static `kernel` parameter is set) |
| `out0` | Output | `(...)` | Convolved output |

## Example

```python
from juniper import Convolution, Gaussian

# Static kernel
conv = Convolution("conv", {
    "kernel": Gaussian({"sigma": (3,), "amplitude": 1, "normalized": True})
})

# Dynamic kernel (from another step)
conv_dyn = Convolution("conv_dyn", {})
kernel_source >> "conv_dyn.kernel"
data_source >> conv_dyn
```
