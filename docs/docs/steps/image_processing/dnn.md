# DNN

Runs a pre-trained VGG16 network (ImageNet weights) on the input image and outputs the activations of a specified layer. The network is only re-evaluated when the input image changes (detected by checksum). Requires the `flaxmodels` package.

**Type:** Static

**Import:** `from juniper.image_processing.DNN import DNN`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `layer` | `str` | Yes | Name of the VGG16 layer whose activations to output |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(H, W, 3)` | RGB image (values 0-255) |
| `out0` | Output | varies | Activation tensor of the specified layer |

## Example

```python
dnn = DNN("dnn", {"layer": "block3_conv3"})
image_source >> dnn
```

**Note:** This step is not JIT-compiled due to the conditional re-evaluation logic.
