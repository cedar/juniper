from functools import partial
import jax

from ...configurables.Step import Step
from ...util import util

def compute_kernel_factory(params,T):
    def compute_kernel(input_mats, buffer, **kwargs):
        output = T.compute(input_mats[util.DEFAULT_INPUT_SLOT], input_mats["in1"])
        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel


class CoordinateTransformation(Step):
    """
    Description
    ---------
    Rigid coordinate transformation of incoming set of 3D vectors.

    Parameters
    ---------
    - FrameGraph : FrameGraph
    - source_frame : str
    - target_frame : str

    Step Input/Output slots
    ---------
    - in0 : jnp.array((N,3))
    - in1 : jnp.array((3,))
    - out0 : jnp.array((N,3))
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["FrameGraph", "source_frame", "target_frame"]
        super().__init__(name, params, mandatory_params)

        self._params = params

        self._frame_graph = self._params["FrameGraph"]
        self._T = self._frame_graph.lookup(source=self._params["source_frame"], target=self._params["target_frame"])

        self.register_input("in1") # input slot for joint angles

        self.compute_kernel = compute_kernel_factory(self._params, self._T)

    