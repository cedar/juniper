from functools import partial
import jax

from .Step import Step
from .. import util

import jax.debug as jdbg


class CoordinateTransformation(Step):
    def __init__(self, name, params):
        mandatory_params = ["FrameGraph", "source_frame", "target_frame"]
        super().__init__(name, params, mandatory_params)

        self._params = params

        self._frame_graph = self._params["FrameGraph"]
        self._T = self._frame_graph.lookup(source=self._params["source_frame"], target=self._params["target_frame"])

        self.register_input("in1") # input slot for joint angles

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_vecs, **kwargs):
        output = self._T.compute(input_vecs[util.DEFAULT_INPUT_SLOT], input_vecs["in1"])
        return {util.DEFAULT_OUTPUT_SLOT: output}
    