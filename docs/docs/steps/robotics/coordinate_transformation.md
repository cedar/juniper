# CoordinateTransformation

Applies a rigid coordinate transformation to a set of 3D vectors, converting them from a source frame to a target frame. Uses a `FrameGraph` to look up the transformation chain. Optionally takes joint angles as a second input to support articulated transforms.

**Type:** Static

**Import:** `from juniper.robotics.steps.CoordinateTransformation import CoordinateTransformation`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `FrameGraph` | `FrameGraph` | Yes | Frame graph object defining coordinate frames and transforms |
| `source_frame` | `str` | Yes | Name of the source coordinate frame |
| `target_frame` | `str` | Yes | Name of the target coordinate frame |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(N, 3)` | Set of 3D vectors in the source frame |
| `in1` | Input | `(3,)` | Joint angles (for articulated transforms) |
| `out0` | Output | `(N, 3)` | Transformed vectors in the target frame |

## Example

```python
from juniper.robotics.configurables.FrameGraph import FrameGraph

fg = FrameGraph(...)  # Define your frame graph
ct = CoordinateTransformation("ct", {
    "FrameGraph": fg,
    "source_frame": "camera",
    "target_frame": "world",
})
vectors >> ct
joint_angles >> "ct.in1"
```
