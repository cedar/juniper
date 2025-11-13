import jax.numpy as jnp

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

def sigmoid_func_singleton(func):
    try:
        sigmoid = SIGMOID_MAP[func]
    except KeyError:
        raise ValueError(
            f"Unknown sigmoid parameter: {func}. "
            f"Supported non-linearities are: {', '.join(SIGMOID_MAP)}"
            )
    return sigmoid

class Sigmoid:
    """
    Description
    ---------
    A wrapper for sigmoid objects. Not intended to be used directly

    Parameters
    ----------
    - type : str(AbsSigmoid, HeavySideSigmoid, ExpSigmoid, LinearSigmoid, SemiLinearSigmoid, LogarithmicSigmoid)
    """
    def __init__(self, type=None):
        self.sigmoid = sigmoid_func_singleton(type)