import jax
from functools import partial
from ..configurables.Step import Step
from .ExpandAxes import ExpandAxes
from .CompressAxes import CompressAxes
from .ReorderAxes import ReorderAxes
from ..util import util

def expand_and_reorder(x, expand_obj, reorder_obj):
    output = expand_obj.compute(x)
    output = reorder_obj.compute({util.DEFAULT_INPUT_SLOT: output[util.DEFAULT_OUTPUT_SLOT]})
    return output
def compress_and_reorder(x, compress_obj, reorder_obj):
    output = compress_obj.compute(x)
    output = reorder_obj.compute({util.DEFAULT_INPUT_SLOT: output[util.DEFAULT_OUTPUT_SLOT]})
    return output
def reorder(x, reorder_obj):
    output = reorder_obj.compute(x)
    return output

def expand_and_reorder_wrapper(expand_obj, reorder_obj):
    return lambda x: expand_and_reorder(x, expand_obj, reorder_obj)
def compress_and_reorder_wrapper(compress_obj, reorder_obj):
    return lambda x: compress_and_reorder(x, compress_obj, reorder_obj)
def reorder_wrapper(reorder_obj):
    return lambda x: reorder(x, reorder_obj)

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

        if out_dim == 1 and self._params["output_shape"][0] == 1:
            out_dim = 0

        if in_dim < out_dim:
            sizes = (self._params["output_shape"][self._params["order"][i]] for i in self._params["axis"])
            expand_obj = ExpandAxes(name + "_expand", {"axis": self._params["axis"], "sizes": sizes})
            reorder_obj = ReorderAxes(name + "_reorder", {"order": self._params["order"]})
            self._compute_func = expand_and_reorder_wrapper(expand_obj, reorder_obj)

        elif in_dim > out_dim:
            reorder_obj = ReorderAxes(name + "_reorder", {"order": self._params["order"]})
            compress_obj = CompressAxes(name + "_compress", {"axis": self._params["axis"], "compression_type": self._params["compression_type"]})
            self._compute_func = compress_and_reorder_wrapper(compress_obj, reorder_obj)

        else:
            reorder_obj = ReorderAxes(name + "_reorder", {"order": self._params["order"]})
            self._compute_func = reorder_wrapper(reorder_obj)


    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):

        output = self._compute_func(input_mats)
        
        return output
    