from ..core.Configurable import Configurable
import jax.numpy as jnp

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
        self._name = "simgoid"
        super().__init__(name="simoid", params=params, mandatory_params=mandatory_params)
        try:
            self.sigmoid = SIGMOID_MAP[self._params["sigmoid"]]
        except KeyError:
            raise ValueError(
                f'Unknown sigmoid parameter: {self._params["sigmoid"]}.'
                f"Supported non-linearities are: {', '.join(SIGMOID_MAP)}"
                )
        

def AbsSigmoid(x, beta, theta):
    return 0.5 * (1.0 + beta * (x - theta) / (1.0 + beta * jnp.abs(x - theta)))

def ExpSigmoid(x, beta, theta):
    return 1.0 / (1.0 + jnp.exp(-beta * (x - theta)))

def HeavySideSigmoid(x, beta, theta):
    return jnp.where(x < theta, 0.0, 1.0)

def LinearSigmoid(x, beta, theta):
    return x * beta - theta

def SemiLinearSigmoid(x, beta, theta):
    return jnp.where(x < theta, 0.0, x * beta - theta)

def LogarithmicSigmoid(x, beta, theta):
    return jnp.log( beta * x - theta )

# Map function names to their corresponding callable
SIGMOID_MAP = {
    "AbsSigmoid": AbsSigmoid,
    "ExpSigmoid": ExpSigmoid,
    "HeavySideSigmoid": HeavySideSigmoid,
    "LinearSigmoid": LinearSigmoid,
    "SemiLinearSigmoid": SemiLinearSigmoid,
    "LogarithmicSigmoid": LogarithmicSigmoid,
}
