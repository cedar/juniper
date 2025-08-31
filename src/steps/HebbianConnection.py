from src.steps.Step import Step
from src import util
from src import util_jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial

def no_reward_gating(passedTime, reward_signal, reward_onset, reward_timer, reward_duration):
    return util_jax.ones((1,)), util_jax.ones((1,)), util_jax.zeros((1,))

def reward_gated(passedTime, reward_signal, reward_onset, reward_timer, reward_duration):
    return reward_signal>0.9, (reward_signal>0.9), 0*reward_signal

def reward_interval(passedTime, reward_signal, reward_onset, reward_timer, reward_duration):
    cond1 = reward_onset == 1
    cond2 = jnp.logical_and(reward_onset == 0, reward_signal > 0.9)

    reward_onset = jnp.select(
        condlist=[cond1, cond2],
        choicelist=[1 - ( reward_timer >= (reward_duration[0] + reward_duration[1]) ), util_jax.ones((1))]
    )
    reward_timer = jnp.select(
        condlist=[cond1, cond2],
        choicelist=[reward_timer, util_jax.zeros((1))]
    )

    reward = reward_onset * jnp.logical_and(
        reward_timer > reward_duration[0],
        reward_timer < (reward_duration[0] + reward_duration[1])
    )
    return reward, reward_onset, reward_timer + passedTime

def instar_learning_rule(target_exp, source_exp, w):
    return (target_exp - w) * source_exp

def outstar_learning_rule(target_exp, source_exp, w):
    return (source_exp - w) * target_exp

def reverse_output(w, target, source_ndim, weight_ndim):
    return jnp.tensordot(w, target, axes=(list(range(source_ndim, weight_ndim)), list(range(0, target.ndim))))

def no_reverse_output(w, target, *_):
    return target * 0

REWARD_MAP = {
    "no_reward": no_reward_gating,
    "reward_gated": reward_gated,
    "reward_interval": reward_interval
}
LEARNING_RULE_MAP = {
    "instar": instar_learning_rule,
    "outstar": outstar_learning_rule
}
BIDIR_MAP = {
    1: reverse_output,
    0: no_reverse_output
}

def make_reward_func(params):
    static_argnames_rew = ['reward_duration']
    try:
        _reward_func = REWARD_MAP[params["reward_duration"]]
    except KeyError:
        raise ValueError(
            f"Unknown reward setting: {params['reward_duration']}. "
            f"Supported settings are: {', '.join(REWARD_MAP)}"
            )
    return partial(jax.jit, static_argnames=static_argnames_rew)(_reward_func)

def make_euler_func(params, static):
    static_argnames_euler = []
    if static:
        static_argnames_euler = ["tau", "tau_decay", "learning_rate", "learning_rule", "bidirectional"]

    # Choose update term based on learning rule
    try:
        learning_rule = LEARNING_RULE_MAP[params["learning_rule"]]
    except KeyError:
        raise ValueError(
            f"Unknown learning rule: {params['learning_rule']}. "
            f"Supported learning rules are: {', '.join(LEARNING_RULE_MAP)}"
            )

    # Choose reverse output calculation
    try:
        output_rev_func = BIDIR_MAP[params["bidirectional"]]
    except KeyError:
        raise ValueError(
            f"Invalid setting for bidirectionality: {params['bidirectional']}. "
            f"Supported functions are: {', '.join(BIDIR_MAP)}"
            )

    def eulerStep(passedTime, prng_key, wheight_mat, source_mat, target_mat, reward, learning_rate, tau, tau_decay):
        # buildup and decay time factors
        timeFactor1 = target_mat * (passedTime / tau)
        timeFactor2 = (1 - target_mat) * (passedTime / tau_decay)
        timeFactor = timeFactor1 + timeFactor2

        # output and reverse output
        output = jnp.tensordot(wheight_mat, source_mat,
            axes=(list(range(0, source_mat.ndim)), list(range(source_mat.ndim))))
        output_rev = output_rev_func(wheight_mat, target_mat, source_mat.ndim, wheight_mat.ndim)

        source_expanded = source_mat.reshape(*source_mat.shape, *([1] * target_mat.ndim))
        target_expanded = target_mat.reshape(*([1] * source_mat.ndim), *target_mat.shape)
        timeFactor_expanded = timeFactor.reshape(*([1] * source_mat.ndim), *target_mat.shape)

        # update wheight matrix
        wheight_mat += reward * timeFactor_expanded * learning_rate * learning_rule(target_expanded, source_expanded, wheight_mat)
        
        return output, output_rev, wheight_mat

    return partial(jax.jit, static_argnames=static_argnames_euler)(eulerStep)

euler_func = None
reward_func = None
def generate_euler_func(static, params):
    global euler_func, reward_func

    # generates euler func based on the geiven params
    if reward_func is None:
        reward_func = make_reward_func(params)
    if euler_func is None:
        euler_func = make_euler_func(params, static)

    return euler_func, reward_func

class HebbianConnection(Step):

    def __init__(self, name, params):
        mandatory_params = ["shape", "target_shape", "tau", "tau_decay", "learning_rate", "learning_rule", "bidirectional", "reward_duration"]

        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self._params["wheight_shape"] = self._params["shape"] + self._params["target_shape"]
        self._params["scalar_shape"] = (1,)

        self._euler_func, self._reward_func = generate_euler_func(util_jax.cfg['euler_step_static_precompile'], self._params)

        self.needs_input_connections = True
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = 3
        self._delta_t = util_jax.get_config()["delta_t"]

        #self.register_input(util.DEFAULT_INPUT_SLOT) # source activation
        self.register_input("in1") # target activaiton
        self.register_input("in2") # reward signal
        self.register_output("out1") # rev_output

        self.register_buffer("wheights", "wheight_shape", save=True) # dynamic wheight parameter
        self.register_buffer("reward_timer", "scalar_shape") # time since reward onset
        self.register_buffer("reward_onset", "scalar_shape") # reward onset

        self.reset()
        

    def compute(self, input_mats, **kwargs):
        if not "prng_key" in kwargs:
            raise Exception("prng_key is a mandatory kwarg to dynamic compute()")

        prng_key = kwargs["prng_key"]
        source_mat = input_mats[util.DEFAULT_INPUT_SLOT]
        target_mat = input_mats["in1"]
        reward_signal = input_mats["in2"]
        
        # Computation
        reward, onset, timer = self._reward_func(self._delta_t, reward_signal, self.buffer["reward_onset"], self.buffer["reward_timer"], self._params["reward_duration"])
        output, output_rev, wheights = self._euler_func(self._delta_t, prng_key, self.buffer["wheights"], source_mat, target_mat, reward, self._params["learning_rate"], self._params["tau"], self._params["tau_decay"])

        # Return output and buffer update
        return {util.DEFAULT_OUTPUT_SLOT: output, "out1": output_rev, 
                "wheights": wheights, "reward_timer": timer, "reward_onset": onset}
    
    def reset(self): # Override default reset, to handle shapes of buffer explicitly.
        self.buffer["wheights"] = util_jax.zeros(self._params["shape"]+self._params["target_shape"])
        self.buffer["reward_timer"] = util_jax.zeros((1,))
        self.buffer["reward_onset"] = util_jax.zeros((1,))
        self.reset_buffer(util.DEFAULT_OUTPUT_SLOT, slot_shape="target_shape")
        self.reset_buffer("out2", slot_shape="shape")
