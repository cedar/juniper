# ViewportCamera

```python
ViewportCamera(name: str, input_shape: tuple, kernel_shape: tuple, viewport_size: tuple, simplified: bool=_simplified, threshold: float=_threshold, move_eps: float=_move_eps, saccade_duration_s: float=_saccade_duration_s, learn_interval_s: float=_learn_interval_s, learn_total_s: float=_learn_total_s)
```

## Description
Selects a viewport-sized RGB crop from an input image based on an activation
map. The step tracks fixation state, emits a one-hot fixation kernel and a
center-of-saccade signal, and can optionally apply learning-mode crop
offsets.

## Parameters
- input_shape : tuple(H,W)
    - Spatial shape of the input image.
- kernel_shape : tuple(H,W)
    - Shape of the activation map and fixation kernel.
- viewport_size : tuple(H,W)
    - Spatial shape of the output crop.
- simplified (optional) : bool
    - If True, directly crops around the activation peak without the full
      saccade state sequence.
    - Default = False
- threshold (optional) : float
    - Activation threshold for selecting a fixation.
    - Default = 0.5
- move_eps (optional) : float
    - Minimum normalized fixation change required to start a new saccade.
    - Default = 0.05
- saccade_duration_s (optional) : float
    - Duration of start/end saccade state phases.
    - Default = 0.020
- learn_interval_s (optional) : float
    - Time between successive learning crop offsets.
    - Default = 0.025
- learn_total_s (optional) : float
    - Total duration of the learning crop sequence.
    - Default = 0.325

## Slots
- in0: jnp.array(input_shape + (3,))
- viewport_center: jnp.array(kernel_shape)
- learn_mode: jnp.array((1,))
- out0: jnp.array(viewport_size + (3,))
- kernel: jnp.array(kernel_shape)
- CoS: jnp.array((1,))

## Import

```python
from juniper import ViewportCamera
```
