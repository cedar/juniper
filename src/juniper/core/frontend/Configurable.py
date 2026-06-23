# Used for parameterizable objects such as steps or kernels.
from ..backend.Exceptions import JuniperConfigurationError
import copy
class Configurable:

    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        self._params = copy.copy(params)
        self._params["name"] = name
        self._local_id = name
        
        for param in mandatory_params:
            if param not in params.keys():
                path_str = self.get_local_circuit_id()
                if hasattr(self, "get_path_str"):
                    path_str = self.get_path_str()
                raise JuniperConfigurationError(f"Parameter {param} is mandatory for objects of type {self.__class__} ({path_str})")
            
    def get_params(self) -> dict:
        return self._params

    def get_local_circuit_id(self) -> str:
        return self._local_id
