# PointCloudToRangeImage

```python
PointCloudToRangeImage(name: str, image_shape: tuple, pan_range: tuple, tilt_range: tuple)
```

## Description
Converts a point cloud into a range image.

## Parameters--
  - pan (azimuth) : [pan_low, pan_high]
  - tilt (polar) :  [tilt_low, tilt_high]
  - image_shape : (n_tilt, n_pan)  [Y, X]

## Slots
- in0: jnp.ndarray(H*W,3)
- out0: jnp.ndarray(image_shape)

## Import

```python
from juniper.robotics import PointCloudToRangeImage
```
