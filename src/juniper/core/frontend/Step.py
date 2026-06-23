import logging
from ..backend.Exceptions import CircuitError

from ...util import util
from ...util import util_jax
from .Element import Element
from .Buffer import Buffer


logger = logging.getLogger(__name__)
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
            raise CircuitError(f"Buffer {buf_id} already registered in step {self.get_path_str()}")
        self.buffer_map[buf_id] = Buffer(self, buf_id, shape, permanent)

    def infer_output_shapes(self, input_specs):
        shape = {}
        for i, out_slot_id in enumerate(self.output_slot_map.keys()):
            in_slot_id = util.DEFAULT_INPUT_SLOT[:-1] + f"{i}"
            if in_slot_id in input_specs.keys() and input_specs[in_slot_id] is not None:
                shape[out_slot_id] = input_specs[in_slot_id][0]
        return shape

    def infer_output_dtypes(self, input_specs):
        dtypes = {slot_id: util_jax.cfg["jdtype"] for slot_id in self.output_slot_map.keys()}
        for i, out_slot_id in enumerate(self.output_slot_map.keys()):
            in_slot_id = util.DEFAULT_INPUT_SLOT[:-1] + f"{i}"
            if in_slot_id in input_specs.keys() and input_specs[in_slot_id][1] is not None:
                dtypes[out_slot_id] = input_specs[in_slot_id][1]
        return dtypes
