import jax.numpy as jnp
from .Configurable import Configurable
from ..math.Sigmoid import SIGMOID_MAP

class Sigmoid(Configurable):
    """
    Description
    ---------
    A wrapper for sigmoid objects. Not intended to be used directly

    Parameters
    ----------
    - type : str(AbsSigmoid, HeavySideSigmoid, ExpSigmoid, LinearSigmoid, SemiLinearSigmoid, LogarithmicSigmoid)
    """
    def __init__(self, params : dict):
        mandatory_params = ["sigmoid"]
        super().__init__(params, mandatory_params)
        try:
            self.sigmoid = SIGMOID_MAP[self._params["sigmoid"]]
        except KeyError:
            raise ValueError(
                f'Unknown sigmoid parameter: {self._params["sigmoid"]}.'
                f"Supported non-linearities are: {', '.join(SIGMOID_MAP)}"
                )