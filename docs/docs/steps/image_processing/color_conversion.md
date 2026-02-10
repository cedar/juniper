# ColorConversion

Converts an RGB image into its three HSV (Hue, Saturation, Value) channels, each output on a separate slot. Input RGB values are expected in the range [0, 255].

**Type:** Static

**Import:** `from juniper import ColorConversion`

## Parameters

None required.

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(H, W, 3)` | RGB image (values 0-255) |
| `out0` | Output | `(H, W)` | Hue channel (0-1) |
| `out1` | Output | `(H, W)` | Saturation channel (0-1) |
| `out2` | Output | `(H, W)` | Value channel (0-1) |

## Example

```python
cc = ColorConversion("cc", {})
image_source >> cc
cc.o0 >> hue_consumer       # Hue
cc.o1 >> sat_consumer       # Saturation
cc.o2 >> val_consumer       # Value
```
