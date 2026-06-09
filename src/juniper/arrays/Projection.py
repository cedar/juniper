from ..core.Step import Step
from ..util import util
import jax.numpy as jnp

def compute_kernel_factory(params, in_dim, out_dim, name):
    if in_dim < out_dim:
        sizes = tuple(params["output_shape"][params["order"][i]] for i in params["axis"])
        def compute_kernel(input_mats, buffer, **kwargs):
            output = jnp.expand_dims(input_mats[util.DEFAULT_INPUT_SLOT], axis=params["axis"])
            for ax, size in zip(params["axis"], sizes):
                output = jnp.repeat(output, size, axis=ax)
            output = jnp.transpose(output, axes=params["order"])
            return {util.DEFAULT_OUTPUT_SLOT: output}

    elif in_dim > out_dim:
        def compute_kernel(input_mats, buffer, **kwargs):
            output = input_mats[util.DEFAULT_INPUT_SLOT]
            output = _compress(output, axis=params["axis"], compression_type=params["compression_type"])
            output = jnp.transpose(output, axes=params["order"])
            return {util.DEFAULT_OUTPUT_SLOT: output}

    else:
        def compute_kernel(input_mats, buffer, **kwargs):
            output = jnp.transpose(input_mats[util.DEFAULT_INPUT_SLOT], axes=params["order"])
            return {util.DEFAULT_OUTPUT_SLOT: output}

    return compute_kernel

def _compress(input, axis, compression_type):
    if compression_type == "Sum":
        return jnp.sum(input, axis=axis)
    if compression_type == "Average":
        return jnp.average(input, axis=axis)
    if compression_type == "Maximum":
        return jnp.max(input, axis=axis)
    if compression_type == "Minimum":
        return jnp.min(input, axis=axis)
    raise ValueError(f"Unknown compression type: {compression_type}")

class Projection(Step):
    """
    Description
    ---------
    The projection step is a combination of an expansion/contraction followed by a reordering of axes. If the input dimensionality is greater than the output a contraction is used.
    Other wise the input is expanded to match the shape of the output. If input and output have the same shape, the axes are only reordered.

    The semantics of the axis parameter changes depending if the input in expanded or contracted. In the contraction case, the axis parameter specifies which of the axes of the input tensor should
    be summed over. In the case of an expansion, the axis parameter specifies the position of the added axis (before reorder). The shape of the added axis will be chosen according to the output_shape.

    Example
    ----------
    - input_shape = (12,11)
    - output_shape = (10,11,12)
    - axis = (2,)
    - order = (2,1,0)
    - The axis parameter says that a new axis will be added to the input array at position 2. The size of this axis is chosen from the output shape. Here the pos 2 axis maps
    onto the 0th axis in the output array. So the new axis will have size 10. After expansion the axis of the expanded input array are reordered to match the output.

    Parameters
    ---------    
    - input_shape : tuple(Nx,Ny,...)
    - output_shape : tuple(Nx,Ny,Nz, ...)
    - axis : tuple(axi,axj,...)
    - order : tuple(axj,axi,...)
    - compression_type : str(Sum,Average,Maximum,Minimum)

    Step Input/Output slots
    ---------
    - in0: jnp.array(input_shape)
    - out0: jnp.ndarray(output_shape)
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["input_shape", "output_shape", "axis", "order", "compression_type"]
        super().__init__(name, params, mandatory_params)
        in_dim = len(self._params["input_shape"])
        out_dim = len(self._params["output_shape"])

        self.compute_kernel = compute_kernel_factory(self._params, in_dim, out_dim, self._name)

    def infer_output_shapes(self, input_specs):
        return {util.DEFAULT_OUTPUT_SLOT: tuple(self._params["output_shape"])}
