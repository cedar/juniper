# Projection

```python
Projection(name: str, input_shape: tuple, output_shape: tuple, axis: tuple, order: tuple, compression_type: str)
```

## Description
The projection step is a combination of an expansion/contraction followed by a reordering of axes. If the input dimensionality is greater than the output a contraction is used.
Other wise the input is expanded to match the shape of the output. If input and output have the same shape, the axes are only reordered.

The semantics of the axis parameter changes depending if the input in expanded or contracted. In the contraction case, the axis parameter specifies which of the axes of the input tensor should
be summed over. In the case of an expansion, the axis parameter specifies the position of the added axis (before reorder). The shape of the added axis will be chosen according to the output_shape.

## Example
- input_shape = (12,11)
- output_shape = (10,11,12)
- axis = (2,)
- order = (2,1,0)
- The axis parameter says that a new axis will be added to the input array at position 2. The size of this axis is chosen from the output shape. Here the pos 2 axis maps
onto the 0th axis in the output array. So the new axis will have size 10. After expansion the axis of the expanded input array are reordered to match the output.

## Parameters    
- input_shape : tuple(Nx,Ny,...)
- output_shape : tuple(Nx,Ny,Nz, ...)
- axis : tuple(axi,axj,...)
- order : tuple(axj,axi,...)
- compression_type : str(Sum,Average,Maximum,Minimum)

## Slots
- in0: jnp.array(input_shape)
- out0: jnp.ndarray(output_shape)

## Import

```python
from juniper import Projection
```
