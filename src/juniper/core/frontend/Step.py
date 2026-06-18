from ...util import util
from ...util import util_jax
from .Element import Element
from .Buffer import Buffer

class Step(Element):
    def __init__(self, name : str, params : dict, mandatory_params : list, is_dynamic : bool = False):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params)
        self.buffer_map : dict[str, Buffer] = {}
        self.is_dynamic = is_dynamic
        self.register_input_slot(util.DEFAULT_INPUT_SLOT)
        self.register_output_slot(util.DEFAULT_OUTPUT_SLOT)

        self.parent.add_element(self)

    def register_buffer(self, buf_id : str, shape : tuple, permanent : bool  = False):
        if buf_id in self.buffer_map.keys():
            raise ValueError(f"Buffer {buf_id} already registered in step {self.get_local_circuit_id()}")
        self.buffer_map[buf_id] = Buffer(self, buf_id, shape, permanent)

    def infer_output_shapes(self, input_specs):
        if "output_shape" in self._params:
            return {util.DEFAULT_OUTPUT_SLOT: tuple(self._params["output_shape"])}
        if "shape" in self._params:
            return {util.DEFAULT_OUTPUT_SLOT: tuple(self._params["shape"])}
        if util.DEFAULT_INPUT_SLOT in input_specs:
            return {util.DEFAULT_OUTPUT_SLOT: input_specs[util.DEFAULT_INPUT_SLOT][0]}
        return {}

    def infer_output_dtypes(self, input_specs):
        dtype = util_jax.cfg["jdtype"]
        if util.DEFAULT_INPUT_SLOT in input_specs and input_specs[util.DEFAULT_INPUT_SLOT][1] is not None:
            dtype = input_specs[util.DEFAULT_INPUT_SLOT][1]
        return {slot_id: dtype for slot_id in self.output_slot_map.keys()}
