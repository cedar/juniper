# Transform

```python
Transform(params)
```

## Description
A wrapper object for a rigid coordinate transformation. The coordinate transformation is given as a function that 
parametarizes the transformation according to the kinematics of the space (i.e. joint states). The function should 
return a 4x4 rigid transformation matrix for any given joint state. The compute function takes a set of vectors and
a joint state to transform the vectors into a new coordinate frame.

This object is used to construct a frame graph, which can be used by the CoordinateTransformation step.

## Parameters-
- M_func: function object 
    - eg. lambda joint_state: jnp.eye(4) (identity)

## Compute
- input_vec : jnp.ndarray((N,3))
- joint_state : jnp.ndarray
- out : jnp.ndarray((N,3))

## Import

```python
from juniper.robotics import Transform
```
