import jax
import jax.numpy as jnp
from functools import partial
from ..configurables.Step import Step
from ..util import util
from typing import Union, Sequence

def nd_norm(
    x: jnp.ndarray,
    ord: Union[int, float, str] = 2,
    axis: Union[int, Sequence[int], None] = None,
    keepdims: bool = True,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Compute an ND norm for JAX arrays over any axis tuple.

    Args:
        x: input array
        ord: order of the norm (supports vector norms like 1, 2, inf, -inf;
             'fro' and other matrix norms fall back to simple flatten behavior).
        axis: axis or sequence of axes to reduce over; defaults to all axes
        keepdims: whether to keep reduced dims for broadcast
        eps: additive epsilon to avoid div-by-zero

    Returns:
        norm with dimensions reduced over `axis` if keepdims=True
    """
    # Normalize axis into a tuple of ints
    if axis is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)

    # L-infinity and -infinity
    if ord == jnp.inf:
        n = jnp.max(jnp.abs(x), axis=axes, keepdims=keepdims)
    elif ord == -jnp.inf:
        n = jnp.min(jnp.abs(x), axis=axes, keepdims=keepdims)

    # L1 norm
    elif ord == 1:
        n = jnp.sum(jnp.abs(x), axis=axes, keepdims=keepdims)

    # L2 norm
    elif ord == 2:
        n = jnp.sqrt(jnp.sum(x * x, axis=axes, keepdims=keepdims))

    # General p-norm
    elif isinstance(ord, (int, float)):
        p = float(ord)
        n = jnp.power(jnp.sum(jnp.power(jnp.abs(x), p), axis=axes, keepdims=keepdims), 1.0 / p)

    # Fallback: flatten over axes
    else:
        flat = jnp.reshape(x, (-1,))
        n = jnp.linalg.norm(flat, ord=ord, keepdims=keepdims)

    return n + eps


NORM_ORDER_MAP = {
    "InfinityNorm": jnp.inf,
    "L1Norm": 1,
    "L2Norm": 2,
}

def compute_kernel_factory(params, ord):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        axis = [i for i in range(len(input.shape))]
        input_norm = nd_norm(input, ord, axis=axis)

        output = input / input_norm
        
        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel

class Normalization(Step):
    """
    Description
    ---------
    Normalizes incoming step using specified norm function.

    Parameters
    ---------    
    - function : str(InfinityNorm, L1Norm, L2Norm)

    Step Input/Output slots
    ---------
    - Input: jnp.ndarray
    - output: jnp.ndarray
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["function"]
        super().__init__(name, params, mandatory_params)
        try:
            self._ord = NORM_ORDER_MAP[self._params["function"]]
        except KeyError:
            raise ValueError(
                f"Unknown function: {self._params['function']}. "
                f"Supported functions are: {', '.join(NORM_ORDER_MAP)}")

        self.compute_kernel = compute_kernel_factory(self._params, self._ord)
    