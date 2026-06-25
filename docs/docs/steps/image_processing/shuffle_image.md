# ShuffleImage

```python
ShuffleImage(name: str, input_shape: tuple, viewport_size: tuple, threshold: float=_threshold, learn_interval_s: float=_learn_interval_s, learn_total_s: float=_learn_total_s)
```

## Description
Crops a viewport-sized RGB image patch from the center of an input image.
During learning, the crop position is shifted through a deterministic
sequence of move the image center accross the viewport.

## Parameters
- input_shape : tuple(H,W)
    - Spatial shape of the input image.
- viewport_size : tuple(H,W)
    - Spatial shape of the output crop.
- threshold (optional) : float
    - Learn-node activation threshold.
    - Default = 0.9
- learn_interval_s (optional) : float
    - Time between successive learning crop offsets.
    - Default = 0.025
- learn_total_s (optional) : float
    - Total duration of the learning crop sequence.
    - Default = 0.325

## Slots
- in0: jnp.array(input_shape + (3,))
- learn_node: jnp.array((1,))
- out0: jnp.array(viewport_size + (3,))

## Import

```python
from juniper import ShuffleImage
```
