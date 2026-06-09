from __future__ import annotations

from typing import Any

from .Step import Step


class Sink(Step):
    def __init__(self, name: str, params: dict, mandatory_params: list, is_dynamic: bool = True):
        super().__init__(name=name, params=params, mandatory_params=mandatory_params, is_dynamic=is_dynamic)
        self.is_sink = True

    def set_data(self, data: Any) -> None:
        raise NotImplementedError(f"Sink {self.get_name()} must implement set_data().")

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass
