# ColorFMap

```python
ColorFMap(name: str, bins: int, saturation_threshold: float=_saturation_threshold, hue_range: int=_hue_range, value_threshold: float=_value_threshold)
```

## Description
Converts hue, saturation and value maps into a 10-channel color feature map.
The first six channels encode red, orange, yellow, green, blue and purple as
one-hot activations. Pixels below the saturation or value threshold are
suppressed.

## Parameters
- bins : int
    - Number of color bins expected by the feature map interface.
- saturation_threshold (optional) : float
    - Minimum saturation required for a pixel to activate a color channel.
    - Default = 0.2
- hue_range (optional) : int
    - Hue range metadata.
    - Default = 360
- value_threshold (optional) : float
    - Minimum value required for a pixel to activate a color channel.
    - Default = 0.2

## Slots
- in0: jnp.array((H,W))
    - Hue channel in [0,1].
- in1: jnp.array((H,W))
    - Saturation channel in [0,1].
- in2: jnp.array((H,W))
    - Value channel in [0,1].
- out0: jnp.array((H,W,10))

## Import

```python
from juniper import ColorFMap
```
