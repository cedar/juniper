# HSV_input

Converts an incoming RGB image into its three HSV channels. Although categorized as a source, this step does accept an input connection. Input RGB values should be in the range [0, 255].

**Type:** Static

**Import:** `from juniper.sources.HSV_input import HSV_input`

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
hsv = HSV_input("hsv", {})
image_source >> hsv
hsv.o0 >> hue_consumer
hsv.o1 >> sat_consumer
hsv.o2 >> val_consumer
```

**Note:** Functionally equivalent to [ColorConversion](../image_processing/color_conversion.md).
