from ..util import util
from ..util import util_jax
from .Element import Element
from .Buffer import Buffer

class Step(Element):
    def __init__(self, name : str, params : dict, mandatory_params : list, is_dynamic : bool = False):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params)
        self.buffer_map : dict[str, Buffer] = {}
        self._max_incoming_connections = {}
        self.is_dynamic = is_dynamic
        self.register_input_slot(util.DEFAULT_INPUT_SLOT)
        self.register_output_slot(util.DEFAULT_OUTPUT_SLOT)

        self.parent.add_element(self)
        
    def register_input_slot(self, slot_id : str, max_incoming_connections : int = 1):
        super().register_input_slot(slot_id, max_incoming_connections)
        self._max_incoming_connections[slot_id] = max_incoming_connections

    def register_buffer(self, buf_id : str, shape : tuple, permanent : bool  = False):
        if buf_id in self.buffer_map.keys():
            raise ValueError(f"Buffer {buf_id} already registered in step {self.get_name()}")
        self.buffer_map[buf_id] = Buffer(self, buf_id, shape, permanent)

    def compile_state(self, input_slots):
        input_specs = {}
        state_updated = False

        input_compile_info_updated, input_specs = self.compile_input_slots(input_slots=input_slots)

        output_compile_info_updated = self.compile_output_slots(input_specs=input_specs)

        buffer_compile_info_updated = self.compile_buffers()

        state_updated = input_compile_info_updated | output_compile_info_updated | buffer_compile_info_updated
        self.check_compiled()
        return state_updated
    

    def compile_output_slots(self, input_specs):
        output_state_updated = False
        output_shapes = self.infer_output_shapes(input_specs)
        output_dtypes = self.infer_output_dtypes(input_specs)
        for slot_id, shape in output_shapes.items():
            if slot_id not in self.output_slot_map or shape is None:
                continue
            slot = self.output_slot_map[slot_id]
            dtype = output_dtypes.get(slot_id, self._default_dtype())
            if slot.shape != shape or slot.dtype != dtype:
                slot.shape = shape
                slot.dtype = dtype
                slot.check_compiled()
                output_state_updated = True
        return output_state_updated

    def compile_buffers(self):
        buffer_updated = False
        for buffer_id, buffer in self.buffer_map.items():
            shape = self._resolve_shape(buffer.shape)
            if shape is None and buffer_id in self.output_slot_map:
                shape = self.output_slot_map[buffer_id].shape
            if shape is None:
                continue
            if buffer.dtype is not None:
                dtype = buffer.dtype
            elif buffer_id in self.output_slot_map:
                dtype = self.output_slot_map[buffer_id].dtype or self._default_dtype()
            else:
                dtype = self._default_dtype()
            if buffer.shape != shape or buffer.dtype != dtype:
                buffer.shape = shape
                buffer.dtype = dtype
                buffer.check_compiled()
                buffer_updated = True
        return buffer_updated
    
    def check_compiled(self):
        for buffer in self.buffer_map.values():
            buffer.check_compiled()
            if not buffer.is_compiled:
                self.is_compiled=False
                return False
        for slot in self.input_slot_map.values():
            slot.check_compiled()
            if not slot.is_compiled and self.needs_input_connections and not self.is_source:
                self.is_compiled=False
                return False
        for slot in self.output_slot_map.values():
            slot.check_compiled()
            if not slot.is_compiled:
                self.is_compiled=False
                return False

        # all slots and buffer are compiled -> step is compiled
        self.is_compiled = True
        return self.is_compiled       

    def _resolve_shape(self, shape):
        if isinstance(shape, str):
            if shape in self._params:
                value = self._params[shape]
                if isinstance(value, int):
                    return (value,)
                return tuple(value)
            return None
        if isinstance(shape, int):
            return (shape,)
        if shape is None:
            return None
        return tuple(shape)

    def infer_output_shapes(self, input_specs):
        if "output_shape" in self._params:
            return {util.DEFAULT_OUTPUT_SLOT: tuple(self._params["output_shape"])}
        if "shape" in self._params:
            return {util.DEFAULT_OUTPUT_SLOT: tuple(self._params["shape"])}
        if util.DEFAULT_INPUT_SLOT in input_specs:
            return {util.DEFAULT_OUTPUT_SLOT: input_specs[util.DEFAULT_INPUT_SLOT][0]}
        return {}

    def infer_output_dtypes(self, input_specs):
        dtype = self._default_dtype()
        if util.DEFAULT_INPUT_SLOT in input_specs and input_specs[util.DEFAULT_INPUT_SLOT][1] is not None:
            dtype = input_specs[util.DEFAULT_INPUT_SLOT][1]
        return {slot_id: dtype for slot_id in self.output_slot_map.keys()}