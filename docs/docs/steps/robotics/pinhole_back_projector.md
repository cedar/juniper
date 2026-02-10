# PinHoleBackProjector

Takes a depth image from a pinhole camera and back-projects each pixel into 3D space, producing a set of 3D position vectors. The camera intrinsics are derived from the focal length and frustum angles.

**Type:** Static

**Import:** `from juniper.robotics.steps.PinHoleBackProjector import PinHoleBackProjector`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `img_shape` | `tuple(H, W)` | Yes | Height and width of the depth image |
| `focal_length` | `float` | Yes | Camera focal length |
| `frustrum_angles` | `tuple(dphi, dtheta)` | Yes | Horizontal and vertical field-of-view angles in degrees |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(H, W)` | Depth image |
| `out0` | Output | `(H*W, 3)` | 3D position vectors |

## Example

```python
proj = PinHoleBackProjector("proj", {
    "img_shape": (480, 640),
    "focal_length": 500.0,
    "frustrum_angles": (60, 45),
})
depth_image >> proj
```
