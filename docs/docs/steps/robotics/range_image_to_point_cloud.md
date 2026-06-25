# RangeImageToPointCloud

```python
RangeImageToPointCloud(name: str, image_shape: tuple, pan_range: tuple, tilt_range: tuple)
```

## Description
Converts a range image (in spherical coordinates) into a point cloud.

## Parameters--
  - pan_range (azimuth) : [pan_low, pan_high]
  - tilt_range (polar) :  [tilt_low, tilt_high]
  - image_shape : (n_tilt, n_pan)  [Y, X]

## Slots
- in0: jnp.ndarray(image_shape)
- out0: jnp.ndarray(H*W,3)

## Import

```python
from juniper.robotics import RangeImageToPointCloud
```
