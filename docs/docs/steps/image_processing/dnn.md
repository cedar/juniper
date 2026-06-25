# DNN

```python
DNN(name: str, layer: str, model_dir: str=_model_dir)
```

## Description
Applies a pretrained VGG16 model to an RGB image and returns the activation
map of a selected convolutional layer. The input image is resized to
(224,224,3) and normalized to [0,1] before it is passed through the model.

## Parameters
- layer : str("4_3", "5_3")
    - VGG16 ReLU layer suffix to read out. The full layer name is built as
      "relu" + layer.
- model_dir (optional) : str
    - Directory used by flaxmodels to find or store VGG16 weights.
    - Default = ".flaxmodels"

## Slots
- in0: jnp.array((H,W,3))
- out0: jnp.array(layer_shape)
    - relu4_3: (28,28,512)
    - relu5_3: (7,7,512)

## Import

```python
from juniper import DNN
```
