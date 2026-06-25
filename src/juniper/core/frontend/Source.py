from __future__ import annotations

import logging
from typing import Any

from .Step import Step



logger = logging.getLogger(__name__)
class Source(Step):
    def __init__(self, name: str, params: dict, mandatory_params: list, is_dynamic: bool = False):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params, is_dynamic=is_dynamic)
        self.is_source = True
        self.needs_input_connections = False
        for slot in self.input_slot_map.values():
            self.parent.connection_map_reversed.pop(slot.get_local_circuit_id(), None)
        self.input_slot_map.clear()

    def get_data(self) -> Any:
        return None

    def set_data(self, data: Any) -> None:
        pass

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass
