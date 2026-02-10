# RangeImageToVectors

Converts a range image (spherical coordinates: pan/azimuth and tilt/elevation) into a set of 3D Cartesian vectors. Each pixel's range value is multiplied by its unit direction vector.

**Type:** Static

**Import:** `from juniper.robotics.steps.RangeImageToVectors import RangeImageToVectors`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_shape` | `tuple(n_tilt, n_pan)` | Yes | Range image dimensions `(Y, X)` |
| `pan_range` | `list[low, high]` | Yes | Azimuth angle range in radians |
| `tilt_range` | `list[low, high]` | Yes | Elevation angle range in radians |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `image_shape` | Range image |
| `out0` | Output | `(n_tilt, n_pan, 3)` | Cartesian position vectors |

## Example

```python
import math
ri2v = RangeImageToVectors("ri2v", {
    "image_shape": (64, 128),
    "pan_range": [-math.pi, math.pi],
    "tilt_range": [-math.pi/4, math.pi/4],
})
range_image >> ri2v
```
