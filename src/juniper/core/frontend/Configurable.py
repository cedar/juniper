# Used for parameterizable objects such as steps or kernels.
from __future__ import annotations

import copy
import inspect
import logging

from ..backend.Exceptions import JuniperConfigurationError, JuniperUserError

logger = logging.getLogger(__name__)

class Configurable:

    def __init__(self, name : str, params : dict | None = None, mandatory_params : dict | None = None):
        if mandatory_params is None:
            mandatory_params = {}
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise JuniperUserError(f"The params argument has to be of type dict but is of type {type(params)} ({name})")

        self._params = copy.copy(params)
        self._params["name"] = name
        self._local_id = name

        for param in mandatory_params:
            if param not in params:
                path_str = self.get_local_circuit_id()
                if hasattr(self, "get_path_str"):
                    path_str = self.get_path_str()
                raise JuniperConfigurationError(f"Parameter {param} is mandatory for objects of type {self.__class__} ({path_str})")
            
        # get the line of where this object was initialized. Used later for error logging
        frame = inspect.currentframe()
        found_outer = False
        while not found_outer:
            func_name = frame.f_code.co_name
            obj = frame.f_locals.get("self", None)
            if func_name == "__init__" and isinstance(obj, Configurable):
                frame = frame.f_back
            else:
                found_outer = True
        self._frame_info = inspect.getframeinfo(frame)


    def get_params(self) -> dict:
        return self._params

    def get_local_circuit_id(self) -> str:
        return self._local_id
