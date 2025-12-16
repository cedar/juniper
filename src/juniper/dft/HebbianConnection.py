from ..configurables.Step import Step
from ..util import util
from ..util import util_jax
import jax.numpy as jnp
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

def make_reward_func(params, static):
    static_argnames_rew = []
    if static:
        static_argnames_rew = ['reward_duration']
    try:
        _reward_func = REWARD_MAP[params["reward_type"]]
    except KeyError:
        raise ValueError(
            f"Unknown reward setting: {params['reward_type']}. "
            f"Supported settings are: {', '.join(REWARD_MAP)}"
            )
    return partial(jax.jit, static_argnames=static_argnames_rew)(_reward_func)

def make_euler_func(params, static):
    static_argnames_euler = []
    if static:
        static_argnames_euler = ["tau", "tau_decay", "learning_rate"]

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
            axes=(list(range(0, source_mat.ndim)), list(range(0,source_mat.ndim))))
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
def euler_func_singleton(static, params):
    global euler_func, reward_func

    # generates euler func based on the geiven params
    if reward_func is None:
        reward_func = make_reward_func(params, static)
    if euler_func is None:
        euler_func = make_euler_func(params, static)

    return euler_func, reward_func

def compute_kernel_factory(params, delta_t):
    _euler_func, _reward_func = euler_func_singleton(util_jax.cfg['euler_step_static_precompile'], params)
    def compute_kernel(input_mats, buffer, **kwargs):
        if "prng_key" not in kwargs:
            raise Exception("prng_key is a mandatory kwarg to dynamic compute()")

        prng_key = kwargs["prng_key"]
        source_mat = input_mats[util.DEFAULT_INPUT_SLOT]
        target_mat = input_mats["in1"]
        reward_signal = input_mats["in2"]
        
        # Computation
        reward, onset, timer = _reward_func(delta_t, reward_signal, buffer["reward_onset"], buffer["reward_timer"], params["reward_duration"])
        output, output_rev, wheights = _euler_func(delta_t, prng_key, buffer["wheights"], source_mat, target_mat, reward, params["learning_rate"], params["tau"], params["tau_decay"])

        # Return output and buffer update
        return {util.DEFAULT_OUTPUT_SLOT: output, "out1": output_rev, 
                "wheights": wheights, "reward_timer": timer, "reward_onset": onset}
    return compute_kernel

class HebbianConnection(Step):
    """
    Description
    ---------
    Implements synaptic connections between the source and target field. Synaptic plasticity 
    is implemented using either the instar or outstar hebbian learning rules, that may be gated 
    by a reward signal. Length and delay of the reward signal can be customized. 

    Parameters
    ---------
    - shape : tuple((Nx,Ny,...))
    - target_shape : tuple((Nx,...))
    - tau (optional) : float
        - Default = 0.01
    - tau_decay (optional) : float
        - Default = 0.1
    - learning_rate (optional) : float
        - Default = 0.1
    - learning_rule (optional) : str("instar", "outstar")
        - Default = instar
    - bidirectional (optional) : bool
        - Default = True
    - reward_type (optional) : str("no_reward", "reward_gated", "reward_interval")
        - Default = no_reward
    - reward_duration (optional) : list[start,stop]
        - Default = [0,1]

    Step Input/Output slots
    ---------
    - in0: jnp.array(shape)
    - in1: jnp.array(target_shape)
    - in2: jnp.array((1,))
    - out0: jnp.array(target_shape)
    - out1: jnp.array(shape)
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["shape", "target_shape"]
        super().__init__(name, params, mandatory_params=mandatory_params, is_dynamic=True)

        if "tau" not in self._params.keys():
            self._params["tau"] = 0.01
        if "tau_decay" not in self._params.keys():
            self._params["tau_decay"] = 0.1
        if "learning_rate" not in self._params.keys():
            self._params["learnig_rate"] = 0.1
        if "learning_rule" not in self._params.keys():
            self._params["learning_rule"] = "instar"
        if "bidirectional" not in self._params.keys():
            self._params["bidirectional"] = True
        if "reward_type" not in self._params.keys():
            self._params["reward_type"] = "no_reward"
        if "reward_duration" not in self._params.keys():
            self._params["reward_duration"] = [0,1]


        self._params["wheight_shape"] = self._params["shape"] + self._params["target_shape"]
        self._params["scalar_shape"] = (1,)
        self._params["reward_duration"] = tuple(self._params["reward_duration"]) 

        self.needs_input_connections = True
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = 3
        self._delta_t = util_jax.get_config()["delta_t"]
        self.compute_kernel = compute_kernel_factory(self._params, self._delta_t)

        #self.register_input(util.DEFAULT_INPUT_SLOT) # source activation
        self.register_input("in1") # target activaiton
        self.register_input("in2") # reward signal
        self.register_output("out1") # rev_output

        self.register_buffer("wheights", "wheight_shape", save=True) # dynamic wheight parameter
        self.register_buffer("reward_timer", "scalar_shape") # time since reward onset
        self.register_buffer("reward_onset", "scalar_shape") # reward onset

        self.reset()
        

    def compute(self, input_mats, **kwargs):
        if "prng_key" not in kwargs:
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
        self.reset_buffer("out1", slot_shape="shape")
        reset_state = {}
        reset_state["wheights"] = self.buffer["wheights"]
        reset_state["reward_timer"] = self.buffer["reward_timer"]
        reset_state["reward_onset"] = self.buffer["reward_onset"]
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self.buffer[util.DEFAULT_OUTPUT_SLOT]
        reset_state["out1"] = self.buffer["out1"]
        return reset_state

