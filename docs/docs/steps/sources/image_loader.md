# ImageLoader

```python
ImageLoader(name: str, image_path: str)
```

Loads an image file and provides it as an array source.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | No input slots |
| Outputs | `out0` image array |

## Import

```python
from juniper import ImageLoader
```

## Notes

- The output shape is inferred from the loaded image.
