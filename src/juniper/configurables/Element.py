from .Connectable import Connectable
from .Slot import Slot
from ..util import util_jax
from typing import Callable
from typing import Optional

class Element(Connectable):
    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        if "." in name:
            raise ValueError(f"Element names cannot contain dots. ({name})")
        
        super().__init__(name=name,params=params, mandatory_params=mandatory_params)
        self.input_slot_map : dict[str, Slot] = {}
        self.output_slot_map : dict[str, Slot] = {}
        self.compute_kernel : Callable[[dict, dict, Optional[dict]], dict] = None
        self.parent = self.parent_circuit

        # Element level meta-data for compiling state-info
        self.is_dynamic = False
        self.is_sink = False
        self.is_source = False
        self.manages_sup_process = False
        self.needs_input_connections = True
        self.input_aggregation = "sum"

        # compiler flag to signal that the internal state has been successfully inferred
        self.is_compiled = False

    def register_output_slot(self, slot_id : str):
        if slot_id in self.output_slot_map.keys():
            raise Exception(f"Output slot {slot_id} already registered in step {self.get_name()}")
        slot = Slot(self, slot_id)
        # Register output slot shortcut
        setattr(self, f"{slot_id}", slot)
        # register slot
        self.output_slot_map[slot_id] = slot

    def register_input_slot(self, slot_id : str, max_incoming_connections : int = 1):
        if slot_id in self.input_slot_map.keys():
            raise Exception(f"Input slot {slot_id} already registered in step {self.get_name()}")
        slot = Slot(self, slot_id, max_incoming_connections)
        # Register input slot shortcut
        setattr(self, f"{slot_id}", slot)
        # register slot
        self.input_slot_map[slot_id] = slot
        # register input slot with parent circuit if not already done so
        if slot.get_name() not in self.parent_circuit.connection_map_reversed.keys():
            self.parent_circuit.connection_map_reversed[slot.get_name()] = []

    def get_max_incoming_connections(self, slot_id : str) -> int:
        slot = self.get_slot(slot_id=slot_id)
        return slot.max_incoming_connections
    
    def get_slot(self, slot_id : str) -> Slot:
        try:
            slot = self.get_input_slot(slot_id)
            return slot
        except Exception:
            try:
                slot = self.get_output_slot(slot_id)
                return slot
            except Exception:
                raise Exception(f"Slot {slot_id} does not exist in step {self.get_name()}")
    
    def get_input_slot(self, slot_id : str) -> Slot:
        if slot_id not in self.input_slot_map.keys():
            raise Exception(f"Slot {slot_id} does not exist in step {self.get_name()}")
        return self.input_slot_map[slot_id]
    
    def get_output_slot(self, slot_id : str) -> Slot:
        if slot_id not in self.output_slot_map.keys():
            raise Exception(f"Slot {slot_id} does not exist in step {self.get_name()}")
        return self.output_slot_map[slot_id]
    
    def compile_state(self, input_slots : dict[str,Slot]) -> bool:
        # Default state inference behavior. No buffer and same shape and dtype of default input slot for default output slot.
        raise NotImplementedError(f"No compile inference behavior specified for element ({self.get_name()})")
        return False
    
    def compile_input_slots(self, input_slots):
        input_specs = {}
        state_updated = False
        for slot_id, slot in self.input_slot_map.items():
            shape, dtype = self._merge_input_slot_compile_info(input_slots.get(slot.get_name(), []))
            if shape is None:
                continue
            if slot.shape != shape or slot.dtype != dtype:
                slot.shape = shape
                slot.dtype = dtype
                slot.check_compiled()
                state_updated = True
            input_specs[slot_id] = (shape, dtype)
        return state_updated, input_specs
    
    def _merge_input_slot_compile_info(self, sources):
        known_sources = [source for source in sources if source.check_compiled()]
        if len(known_sources) == 0:
            return None, None

        shape = known_sources[0].shape
        dtype = known_sources[0].dtype or self._default_dtype()
        for source in known_sources[1:]:
            source_shape = source.shape
            if source_shape != shape:
                if self._is_scalar_shape(shape):
                    shape = source_shape
                elif not self._is_scalar_shape(source_shape):
                    raise ValueError(f"Step {self.get_name()} received incompatible input shapes {shape} and {source_shape}")
            if source.dtype is not None:
                dtype = source.dtype
        return shape, dtype

    def _default_dtype(self):
        return util_jax.cfg["jdtype"]
    
    def _is_scalar_shape(self, shape):
        return shape == () or shape == (1,)