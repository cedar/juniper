import logging
from ..core.frontend.Step import Step
from ..util import util, util_jax
import jax
import jax.numpy as jnp
from functools import partial
from ..core.backend.Exceptions import JuniperUserError


logger = logging.getLogger(__name__)
def make_euler_bcm_func(params, static):
    static_argnames = []
    if static:
        static_argnames = [
            "learning_rate",
            "tau_weights",
            "tau_theta",
            "min_theta",
            "use_fixed_theta",
            "fixed_theta",
            "theta_eps",
            "norm_target",
            "norm_rate",
            "safeguard_thr"
        ]

    def eulerStepBCM(
        dt,                
        w,                
        theta,             
        source,            
        target,            
        reward_signal,     
        learning_rate,
        tau_weights,       
        tau_theta,         
        min_theta,
        use_fixed_theta,
        fixed_theta,
        theta_eps,
        norm_target,
        norm_rate,
        safeguard_thr
    ):
        reward_on = (reward_signal.reshape(-1)[0] > 0.5)

        out = jnp.einsum("xyfd,xyf->xyd", w, source)
        
        learn_mask = jnp.all((target[1:-1,1:-1,:] > safeguard_thr), axis=(0, 1)) 
        active_k = jnp.argmax(learn_mask)
        active_mask = learn_mask[active_k] 

        timeFactor_w = dt / tau_weights
        theta_safe = jnp.maximum(jnp.maximum(theta, min_theta), theta_eps)

        x = source[..., None]           
        y = target[:, :, None, :]     
        th = theta_safe[:, :, None, :] 

        learning_rate = -norm_rate * learning_rate * (jnp.max(out[:, :, active_k]) - norm_target)
        dw_bcm = (timeFactor_w * learning_rate * y * (y - th) * (x / th)) * jnp.zeros((w.shape[-1],)).at[active_k].set(active_mask) 
        w = w + reward_on * (dw_bcm)
        w = jnp.maximum(w, 0)

        def update_theta(th_in):
            timeFactor_t = dt / tau_theta
            th_new = th_in + timeFactor_t * (target * target - th_in)
            return jnp.maximum(th_new, min_theta)
        theta = jax.lax.cond(
            use_fixed_theta,
            lambda _: jnp.ones_like(theta) * fixed_theta,
            lambda _: update_theta(theta),
            operand=None,
        )
        return out, w, theta
    return eulerStepBCM

_euler_bcm_singleton = None

def euler_bcm_singleton(static, params):
    global _euler_bcm_singleton
    if _euler_bcm_singleton is None:
        _euler_bcm_singleton = make_euler_bcm_func(params, static)
    return _euler_bcm_singleton

def compute_kernel_factory(params, delta_t):
    euler = euler_bcm_singleton(util_jax.cfg["euler_step_static_precompile"], params)

    def compute_kernel(input_mats, buffer, **kwargs):
        source = input_mats[util.DEFAULT_INPUT_SLOT]
        target = input_mats["in1"]
        reward = input_mats["in2"]
        out, w, theta = euler(
            delta_t,
            buffer["wheights"],
            buffer["theta"],
            source,
            target,
            reward,
            params["learning_rate"],
            params["tau_weights"],
            params["tau_theta"],
            params["min_theta"],
            bool(params["use_fixed_theta"]),
            float(params["fixed_theta"]),
            float(params["theta_eps"]),
            params["norm_target"],
            params["norm_rate"],
            params["safeguard_thr"]
        )
        return {
            util.DEFAULT_OUTPUT_SLOT: out,
            "wheights": w,
            "theta": theta,
        }
    return compute_kernel


class BCMConnection(Step):
    """
    Description
    ---------
    Implements a BCM-style synaptic connection between source and target fields.
    Weights are learned from source and target activity when the reward input is
    active. The target trace theta is either updated dynamically or clamped to a
    fixed value, depending on the use_fixed_theta setting.

    Parameters
    ---------
    - source_shape : tuple((Nx,Ny,Nf))
        - Source field shape.
    - target_shape : tuple((Nx,Ny,Nd))
        - Target field shape. The first two dimensions must match shape.
    - tau_weights (optional) : float
        - Default = 1.0
    - tau_theta (optional) : float
        - Default = 1.0
    - learning_rate (optional) : float
        - Default = 0.1
    - min_theta (optional) : float
        - Default = 0.0
    - use_fixed_theta (optional) : bool
        - Default = True
    - fixed_theta (optional) : float
        - Default = 0.25
    - norm_target (optional) : float
        - Default = 0.0
    - norm_rate (optional) : float
        - Default = 0.0
    - safeguard_thr (optional) : float
        - Default = -1.0
    - theta_eps (optional) : float
        - Default = 1e-6

    Step Input/Output slots
    ---------
    - in0: jnp.array(shape)
    - in1: jnp.array(target_shape)
    - in2: jnp.array((1,))
    - out0: jnp.array(target_shape)
    """

    _tau_weights = 1.0
    _tau_theta = 1.0
    _learning_rate = 0.1
    _min_theta = 0.0
    _use_fixed_theta = True
    _fixed_theta = 0.25
    _norm_target = 0.0
    _norm_rate = 0.0
    _safeguard_thr = -1.0
    _theta_eps = 1e-6
    def __init__(
            self,
            name : str,
            source_shape : tuple,
            target_shape : tuple,
            tau_weights : float = _tau_weights,
            tau_theta : float = _tau_theta,
            learning_rate : float = _learning_rate,
            min_theta : float = _min_theta,
            use_fixed_theta : bool = _use_fixed_theta,
            fixed_theta : float = _fixed_theta,
            norm_target : float = _norm_target,
            norm_rate : float = _norm_rate,
            safeguard_thr : float = _safeguard_thr,
            theta_eps : float = _theta_eps):
        params = locals().copy()
        mandatory = [
            "source_shape",
            "target_shape",
        ]
        super().__init__(name, params, mandatory_params=mandatory, is_dynamic=True)
        sx, sy, sf = self._params["source_shape"]
        tx, ty, td = self._params["target_shape"]
        if (sx, sy) != (tx, ty):
            raise JuniperUserError(f"BCMConnection requires source and target to match in first two dims (X,Y) ({self.get_path_str()}).")
        self._params["wheight_shape"] = (sx, sy, sf, td)
        self._params["scalar_shape"] = (1,)
        self._params["theta_eps"] = float(self._params.get("theta_eps", 1e-6))
        self._delta_t = float(util_jax.get_config()["delta_t"])
        self.register_input_slot("in1")  
        self.register_input_slot("in2")  
        self.register_buffer("wheights", self._params["wheight_shape"], permanent=True)
        self.register_buffer("theta", self._params["target_shape"], permanent=True)

        self.compute_kernel = compute_kernel_factory(self._params, self._delta_t)

    def infer_output_shapes(self, input_specs):
        return {util.DEFAULT_OUTPUT_SLOT: tuple(self._params["target_shape"])}
