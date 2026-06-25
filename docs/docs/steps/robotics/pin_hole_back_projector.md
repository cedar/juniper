# PinHoleBackProjector

```python
PinHoleBackProjector(name: str, img_shape: tuple, focal_length: float, frustrum_angles: tuple)
```

## Description
Takes the depth image of a pinhole camera as input and transforms the image into a point cloud.

## Parameters    
- img_shape : tuple(H,W)
- focal_length : float
- frustrum_angles : tuple(dphi, dtheta)

## Slots
- Input: jnp.array(img_shape)
- output: jnp.ndarray(H*W,3)

## Import

```python
from juniper.robotics import PinHoleBackProjector
```
