from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..math.LateralKernel import LateralKernel

from ..core.frontend.Step import Step
from ..util import util
from ..util import util_jax
import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from ..math.Sigmoid import Sigmoid

import logging
logger = logging.getLogger(__name__)

# This singleton construct is needed as we need to specify the static_argnames in the compiler directive depending on the user input
# euler step computation
@partial(jax.jit, static_argnames=["passedTime",  "resting_level", "global_inhibition", "beta", "theta", "tau", "input_noise_gain", "sigmoid", "convolve"])
def eulerStep(passedTime, input_mat, u_activation, prng_key, resting_level, global_inhibition, beta, theta, tau, input_noise_gain, sigmoid, convolve):
    sigmoided_u = sigmoid(u_activation, beta, theta) # Could be optimized, we don't need this sigmoid computation if we pass the value of the output buffer to this function (which effectively is the sigmoided_u)
    lateral_interaction = convolve(sigmoided_u)

    sum_sigmoided_u = jnp.sum(sigmoided_u)

    d_u = -u_activation + resting_level + lateral_interaction + global_inhibition * sum_sigmoided_u + input_mat

    input_noise = jax.random.normal(prng_key, input_mat.shape)
    u_activation += (passedTime / tau) * d_u + ((jnp.sqrt(passedTime*1000) / tau/1000)) * input_noise_gain * input_noise

    sigmoided_u = sigmoid(u_activation, beta, theta)

    return sigmoided_u, u_activation


def compute_kernel_factory(passedTime, resting_level, global_inhibition, beta, theta, tau, input_noise_gain, sigmoid, convolve):
    def compute_kernel(input_mats, buffer, **kwargs):
        sigmoided_u, u_activation = eulerStep(passedTime, input_mats[util.DEFAULT_INPUT_SLOT], buffer["activation"], kwargs["prng_key"], resting_level, global_inhibition, beta, 
                                                          theta, tau, input_noise_gain, sigmoid, convolve)
        return {util.DEFAULT_OUTPUT_SLOT: sigmoided_u, "activation":u_activation}
    return compute_kernel

class NeuralField(Step):
    """
    Description
    ---------
    Neural Field step. 

    Parameters
    ---------    
    - shape : tuple(Nx,Ny,...)
    - sigmoid (optional) : str(AbsSigmoid, HeavySideSigmoid, ExpSigmoid, LinearSigmoid, SemiLinearSigmoid, LogarithmicSigmoid)
    - beta (optional) : float
    - theta (optional) : float
    - resting_level (optional) : float
    - global_inhibition (optional) : float
    - input_noise_gain (optional) : float
    - tau (optional) [ms] : float
    - LateralKernel (optional) : LateralKernel or Gaussian

    Step Input/Output slots
    ---------
    - Input: jnp.ndarray(shape)
    - output: jnp.ndarray(shape)
    """
    # Default params
    _sigmoid = "AbsSigmoid"
    _beta = 100
    _theta = 0
    _resting_level = -5
    _global_inhibition = 0
    _input_noise_gain = 0
    _tau = util_jax.get_config()["delta_t"] * 10
    _lateral_kernel = None
    def __init__(self, 
                 name : str, 
                 shape : tuple[int, ...], 
                 sigmoid : str = _sigmoid,
                 beta : int = _beta,
                 theta : float = _theta,
                 resting_level : float = _resting_level,
                 global_inhibition : float = _global_inhibition,
                 input_noise_gain : float = 0,
                 tau : float = _tau,
                 lateral_kernel : LateralKernel | None = _lateral_kernel
                 ):
        params = locals().copy()
        mandatory_params = ["shape"]
        super().__init__(name, params, mandatory_params, is_dynamic=True)

        self.needs_input_connections = False
        self.set_max_incoming_connections(util.DEFAULT_INPUT_SLOT, np.inf)
        delta_t = util_jax.get_config()["delta_t"]

        if lateral_kernel is None:
            def lateral_kernel_convolve(x):
                return x*0
        else:
            lateral_kernel_convolve = lateral_kernel.gen_convolve_func()

        sigmoid = Sigmoid({"sigmoid":sigmoid}).sigmoid

        self.compute_kernel = compute_kernel_factory(delta_t, resting_level, global_inhibition, beta, theta, tau, input_noise_gain, 
                                                       sigmoid, lateral_kernel_convolve)
        
        self.register_buffer("activation", self._params["shape"])

    def infer_output_shapes(self, input_specs):
        return {util.DEFAULT_OUTPUT_SLOT: self._params["shape"]}