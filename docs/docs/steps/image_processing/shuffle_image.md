# ShuffleImage

```python
ShuffleImage(
    name: str,
    input_shape: tuple,
    viewport_size: tuple,
    threshold: float = 0.9,
    learn_interval_s: float = 0.025,
    learn_total_s: float = 0.325,
)
```

Creates a shuffled viewport image for learning or data augmentation workflows.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` image; `learn_node` control signal |
| Outputs | `out0` viewport-sized image |

## Import

```python
from juniper import ShuffleImage
```

## Notes

- Dynamic buffers track learning timing.
