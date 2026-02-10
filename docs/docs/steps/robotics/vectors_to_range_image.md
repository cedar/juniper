# VectorsToRangeImage

Projects a set of 3D Cartesian vectors into a 2D spherical range image. Vectors are converted to spherical coordinates (azimuth, elevation, range) and binned into the image grid. When multiple vectors fall into the same pixel, the closest range is kept.

**Type:** Static

**Import:** `from juniper.robotics.steps.VectorsToRangeImage import VectorsToRangeImage`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_shape` | `tuple(n_tilt, n_pan)` | Yes | Range image dimensions `(Y, X)` |
| `pan_range` | `list[low, high]` | Yes | Azimuth angle range in radians |
| `tilt_range` | `list[low, high]` | Yes | Elevation angle range in radians |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(N, 3)` | 3D Cartesian vectors |
| `out0` | Output | `image_shape` | Range image (pixels with no data are set to 0) |

## Example

```python
import math
v2ri = VectorsToRangeImage("v2ri", {
    "image_shape": (64, 128),
    "pan_range": [-math.pi, math.pi],
    "tilt_range": [-math.pi/4, math.pi/4],
})
vectors >> v2ri
```
