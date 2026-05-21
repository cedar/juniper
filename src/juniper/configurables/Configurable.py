# Used for parameterizable objects such as steps or kernels.
import copy
class Configurable:

    def __init__(self, name : str, params : dict = {}, mandatory_params : dict = {}):
        self._params = copy.copy(params)
        self._params["name"] = name
        self._name = name
        
        for param in mandatory_params:
            if param not in params.keys():
                raise Exception(f"Parameter {param} is mandatory ({self.get_name()})")
            
    def get_params(self) -> dict:
        return self._params

    def get_name(self) -> str:
        return self.name