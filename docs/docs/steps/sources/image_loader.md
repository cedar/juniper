# ImageLoader

Loads an image file from disk and outputs it as a float32 array. The image is re-read on each tick.

**Type:** Static (Source)

**Import:** `from juniper import ImageLoader`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_path` | `str` | Yes | Path to the image file |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `out0` | Output | `(H, W, C)` | Image as float32 array |

## Example

```python
img = ImageLoader("img", {"image_path": "data/scene.png"})
img >> color_conversion
```
