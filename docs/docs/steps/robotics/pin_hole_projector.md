# PinHoleProjector

```python
PinHoleProjector(name: str, img_shape: tuple, focal_length: float, frustrum_angles: tuple)
```

## Description
Takes a point cloud and transforms it into a depth image of a pinhole depth camera.

## Parameters    
- img_shape : tuple(H,W)
- focal_length : float
- frustrum_angles : tuple(dphi, dtheta)

## Slots
- Input: jnp.array(img_shape)
- output: jnp.ndarray(H*W,3)

## Import

```python
from juniper.robotics import PinHoleProjector
```
