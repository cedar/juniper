# ViewportCamera

```python
ViewportCamera(
    name: str,
    input_shape: tuple,
    kernel_shape: tuple,
    viewport_size: tuple,
    simplified: bool = False,
    threshold: float = 0.5,
    move_eps: float = 0.05,
    saccade_duration_s: float = 0.020,
    learn_interval_s: float = 0.025,
    learn_total_s: float = 0.325,
)
```

Extracts a viewport from an image based on an activation field and optional saccade-like timing.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` image; `viewport_center` activation/control input; `learn_mode` control input |
| Outputs | `out0` viewport image; `kernel` fixation kernel; `CoS` change-of-state signal |

## Import

```python
from juniper import ViewportCamera
```

## Notes

- Dynamic buffers track fixation, movement, and timing state.
