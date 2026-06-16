from ..core.Step import Step
from ..util import util
from ..util import util_jax
import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from ..math.Sigmoid import Sigmoid

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
    - sigmoid : str(AbsSigmoid, HeavySideSigmoid, ExpSigmoid, LinearSigmoid, SemiLinearSigmoid, LogarithmicSigmoid)
    - beta : float
    - theta : float
    - resting_level : float
    - global_inhibition : float
    - input_noise_gain : float
    - tau [ms] : float
    - LateralKernel (optional) : LateralKernel or Gaussian

    Step Input/Output slots
    ---------
    - Input: jnp.ndarray(shape)
    - output: jnp.ndarray(shape)
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["shape", "sigmoid", "beta", "theta", "resting_level", "global_inhibition", "input_noise_gain", "tau"]
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.needs_input_connections = False
        self.set_max_incoming_connections(util.DEFAULT_INPUT_SLOT, np.inf)
        self._delta_t = util_jax.get_config()["delta_t"]

        if "LateralKernel" not in self._params:
            self._lateral_kernel_convolve = lambda x: x*0
        else:
            self._lateral_kernel_convolve = self._params["LateralKernel"].gen_convolve_func()

        self.sigmoid = Sigmoid({"sigmoid":self._params["sigmoid"]}).sigmoid

        self.compute_kernel = compute_kernel_factory(self._delta_t, self._params["resting_level"], self._params["global_inhibition"], 
                                                       self._params["beta"], self._params["theta"], self._params["tau"], self._params["input_noise_gain"], 
                                                       self.sigmoid, self._lateral_kernel_convolve)
        
        self.register_buffer("activation", self._params["shape"])
