from ..configurables.Step import Step
from ..util import util, util_jax
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

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
    ):
        reward_on = (reward_signal.reshape(-1)[0] > 0.5)
        out = jnp.einsum("xyfd,xyf->xyd", w, source)
        timeFactor_w = dt / tau_weights
        theta_safe = jnp.maximum(theta, min_theta)
        theta_safe = jnp.maximum(theta_safe, theta_eps)
        x = source[..., None]
        y = target[:, :, None, :]
        th = theta_safe[:, :, None, :]
        dw = timeFactor_w * learning_rate * y * (y - th) * (x / th)
        w = w + reward_on * dw
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
    return partial(jax.jit, static_argnames=static_argnames)(eulerStepBCM)

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
        )
        return {
            util.DEFAULT_OUTPUT_SLOT: out,
            "wheights": w,
            "theta": theta,
        }
    return compute_kernel


class BCMConnection(Step):

    def __init__(self, name, params):
        mandatory = [
            "shape",
            "target_shape",
            "tau_weights",
            "tau_theta",
            "learning_rate",
            "min_theta",
            "use_fixed_theta",
            "fixed_theta",
        ]
        super().__init__(name, params, mandatory_params=mandatory, is_dynamic=True)
        sx, sy, sf = self._params["shape"]
        tx, ty, td = self._params["target_shape"]
        if (sx, sy) != (tx, ty):
            raise ValueError("BCMConnection requires source and target to match in first two dims (X,Y).")
        self._params["wheight_shape"] = (sx, sy, sf, td)
        self._params["scalar_shape"] = (1,)
        self._params["theta_eps"] = float(self._params.get("theta_eps", 1e-6))
        self._delta_t = float(util_jax.get_config()["delta_t"])
        self.register_input("in1")  
        self.register_input("in2")  
        self.register_buffer("wheights", "wheight_shape", save=True)
        self.register_buffer("theta", "target_shape", save=True)
        self.cpu_buffer = {}

        self.compute_kernel = compute_kernel_factory(self._params, self._delta_t)

        self.reset()

    def reset(self):
        self.buffer["wheights"] = util_jax.zeros(self._params["wheight_shape"])
        init_theta = jnp.float32(self._params["fixed_theta"])
        self.buffer["theta"] = util_jax.ones(self._params["target_shape"]) * init_theta
        self.reset_buffer(util.DEFAULT_OUTPUT_SLOT, slot_shape="target_shape")
        self.reset_buffer("out1", slot_shape="shape")
        self.cpu_buffer["wheights"] = np.array(self.buffer["wheights"])
        self.cpu_buffer["theta"] = np.array(self.buffer["theta"])
        reset_state = {}
        reset_state["wheights"] = self.buffer["wheights"]
        reset_state["theta"] = self.buffer["theta"]
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self.buffer[util.DEFAULT_OUTPUT_SLOT]
        return reset_state
