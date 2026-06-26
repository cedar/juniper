# Transform

```python
Transform({"M_func": callable})
```

Wraps a function that maps a joint state to a 4x4 homogeneous transformation matrix. `CoordinateTransformation` uses `Transform` objects through a `FrameGraph`.

## Example

```python
import jax.numpy as jnp
from juniper.robotics import Transform

identity = Transform({"M_func": lambda joint_state: jnp.eye(4)})
```

The transform kernel expects points with shape `(N, 3)` and returns transformed points with shape `(N, 3)`.
