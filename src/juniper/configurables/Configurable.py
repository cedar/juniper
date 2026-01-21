# Used for parameterizable objects such as steps or kernels.
import copy
class Configurable:

    def __init__(self, params, mandatory_params):
        self._params = copy.copy(params)
        self._params["name"] = self._name
        #self._params = params
        for param in mandatory_params:
            if not param in params:
                raise ValueError(f"Parameter {param} is mandatory.")