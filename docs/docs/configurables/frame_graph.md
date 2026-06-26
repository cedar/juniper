# FrameGraph

```python
FrameGraph(params={})
```

Stores named coordinate frames and directed transform edges. `CoordinateTransformation` looks up the transform from a source frame to a target frame and applies it during simulation.

## Methods

| Method | Description |
|--------|-------------|
| `add_edge(source, target, transform)` | Register a transform between two named frames. |
| `lookup(source, target)` | Return a direct, inverse, or composed transform between frames. |

## Example

```python
import jax.numpy as jnp
from juniper.robotics import FrameGraph, Transform

frames = FrameGraph({})
frames.add_edge("camera", "world", Transform({"M_func": lambda q: jnp.eye(4)}))
```
