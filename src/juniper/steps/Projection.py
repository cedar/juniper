import jax
import jax.numpy as jnp
from functools import partial
from .Step import Step
from .ExpandAxes import ExpandAxes
from .CompressAxes import CompressAxes
from .ReorderAxes import ReorderAxes
from .. import util

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

    def __init__(self, name, params):
        mandatory_params = ["input_shape", "output_shape", "axis", "order", "compression_type"]
        super().__init__(name, params, mandatory_params)
        in_dim = len(self._params["input_shape"])
        out_dim = len(self._params["output_shape"])

        if out_dim == 1 and self._params["output_shape"][0] == 1:
            out_dim = 0

        if in_dim < out_dim:
            sizes = (self._params["output_shape"][i] for i in self._params["axis"])
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
    