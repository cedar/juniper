# DNN

```python
DNN(name: str, layer: str, model_dir: str = ".flaxmodels")
```

Runs an image through a pretrained neural network layer.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` image-like input |
| Outputs | `out0` layer activation |

## Import

```python
from juniper import DNN
```

## Notes

- Requires flax and flaxmodels.
- Model files are read from `model_dir`; the constructor may prompt before downloading missing data.
