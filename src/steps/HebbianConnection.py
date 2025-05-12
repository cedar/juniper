from src.steps.Step import Step
from src import util
from src import util_jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial

euler_func = None
reward_func = None
def generate_euler_func(static, params):
    global euler_func
    global reward_func

    static_argnames_rew = ['reward_duration']
    static_argnames_euler = []
    if static:
        static_argnames_euler = ["tau", "tau_decay", "learning_rate", "learning_rule", "bidirectional"]

    print("static_argnames", static_argnames_euler)
    if reward_func is not None:
        pass
    else:
        if params["reward_duration"] == "no_reward":
            @partial(jax.jit, static_argnames=static_argnames_rew)
            def _reward_func(passedTime, reward_signal, reward_onset, reward_timer, reward_duration): 
                return util_jax.ones((1,)), util_jax.ones((1,)), util_jax.zeros((1,))
            reward_func = _reward_func
        elif params["reward_duration"] == "reward_gated":
            @partial(jax.jit, static_argnames=static_argnames_rew)
            def _reward_func(passedTime, reward_signal, reward_onset, reward_timer, reward_duration):
                return reward_signal>0.9, (reward_signal>0.9), 0*reward_signal
            reward_func = _reward_func
        elif len(params["reward_duration"]) == 2:
            @partial(jax.jit, static_argnames=static_argnames_rew)
            def _reward_func(passedTime, reward_signal, reward_onset, reward_timer, reward_duration):
                cond1 = reward_onset == 1
                cond2 = jnp.logical_and(reward_onset == 0, reward_signal > 0.9)

                reward_onset = jnp.select(condlist=[cond1, cond2],
                           choicelist=[1 - ( reward_timer >= (reward_duration[0] + reward_duration[1]) ), util_jax.ones((1))])
                reward_timer = jnp.select(condlist=[cond1, cond2],
                           choicelist=[reward_timer, util_jax.zeros((1))])

                reward = reward_onset * jnp.logical_and(reward_timer > reward_duration[0], reward_timer < (reward_duration[0] + reward_duration[1]))
                return reward, reward_onset, reward_timer + passedTime
            reward_func = _reward_func
        else:
            raise ValueError("Invalid reward duration specified. Must be 'no_reward', 'reward_gated', or a tuple of (onset, duration).")

    if euler_func is not None:
        pass  
    else:
        if params["learning_rule"] == "instar":
            if params["bidirectional"]:
                @partial(jax.jit, static_argnames=static_argnames_euler)
                def eulerStep(passedTime, prng_key, wheight_mat, source_mat, target_mat, reward, learning_rate, tau, tau_decay):
                    timeFactor1 = target_mat * (passedTime / tau)
                    timeFactor2 = ( 1 - target_mat ) * (passedTime / tau_decay)
                    timeFactor = timeFactor1 + timeFactor2

                    output = jnp.tensordot(wheight_mat, source_mat, axes=(list(range(0,source_mat.ndim)), list(range(source_mat.ndim))))
                    output_rev = jnp.tensordot(wheight_mat, target_mat, axes=(list(range(source_mat.ndim,wheight_mat.ndim)), list(range(0, target_mat.ndim))))

                    source_expanded = source_mat.reshape(*source_mat.shape, *([1] * target_mat.ndim))
                    target_expanded = target_mat.reshape(*([1] * source_mat.ndim), *target_mat.shape)
                    timeFactor_expanded = timeFactor.reshape(*([1] * source_mat.ndim), *target_mat.shape)
                    
                    wheight_mat += reward * timeFactor_expanded * learning_rate * ( ( target_expanded - wheight_mat ) * source_expanded )
                
                    return output, output_rev, wheight_mat
                euler_func = eulerStep
            else:
                @partial(jax.jit, static_argnames=static_argnames_euler)
                def eulerStep(passedTime, prng_key, wheight_mat, source_mat, target_mat, reward, learning_rate, tau, tau_decay):
                    timeFactor1 = target_mat * (passedTime / tau)
                    timeFactor2 = ( 1 - target_mat ) * (passedTime / tau_decay)
                    timeFactor = timeFactor1 + timeFactor2

                    output = jnp.tensordot(wheight_mat, source_mat, axes=(list(range(0,source_mat.ndim)), list(range(source_mat.ndim))))
                    output_rev = source_mat*0
 
                    source_expanded = source_mat.reshape(*source_mat.shape, *([1] * target_mat.ndim))
                    target_expanded = target_mat.reshape(*([1] * source_mat.ndim), *target_mat.shape)
                    timeFactor_expanded = timeFactor.reshape(*([1] * source_mat.ndim), *target_mat.shape)

                    wheight_mat += reward * timeFactor_expanded * learning_rate * ( ( target_expanded - wheight_mat ) * source_expanded )
                
                    return output, output_rev, wheight_mat
                euler_func = eulerStep
        elif params["learning_rule"] == "outstar":
            if params["bidirectional"]:
                @partial(jax.jit, static_argnames=static_argnames_euler)
                def eulerStep(passedTime, prng_key, wheight_mat, source_mat, target_mat, reward, learning_rate, tau, tau_decay):
                    timeFactor1 = target_mat * (passedTime / tau)
                    timeFactor2 = ( 1 - target_mat ) * (passedTime / tau_decay)
                    timeFactor = timeFactor1 + timeFactor2

                    output = jnp.tensordot(wheight_mat, source_mat, axes=(list(range(0,source_mat.ndim)), list(range(source_mat.ndim))))
                    output_rev = jnp.tensordot(wheight_mat, target_mat, axes=(list(range(source_mat.ndim,wheight_mat.ndim)), list(range(0, target_mat.ndim))))

                    source_expanded = source_mat.reshape(*source_mat.shape, *([1] * target_mat.ndim))
                    target_expanded = target_mat.reshape(*([1] * source_mat.ndim), *target_mat.shape)
                    timeFactor_expanded = timeFactor.reshape(*([1] * source_mat.ndim), *target_mat.shape)

                    wheight_mat += reward * timeFactor_expanded * learning_rate * ( ( source_expanded - wheight_mat ) * target_expanded )

                    return output, output_rev, wheight_mat
                euler_func = eulerStep
            else:
                @partial(jax.jit, static_argnames=static_argnames_euler)
                def eulerStep(passedTime, prng_key, wheight_mat, source_mat, target_mat, reward, learning_rate, tau, tau_decay):
                    timeFactor1 = target_mat * (passedTime / tau)
                    timeFactor2 = ( 1 - target_mat ) * (passedTime / tau_decay)
                    timeFactor = timeFactor1 + timeFactor2
                    
                    output = jnp.tensordot(wheight_mat, source_mat, axes=(list(range(0,source_mat.ndim)), list(range(source_mat.ndim))))
                    output_rev = source_mat*0

                    source_expanded = source_mat.reshape(*source_mat.shape, *([1] * target_mat.ndim))
                    target_expanded = target_mat.reshape(*([1] * source_mat.ndim), *target_mat.shape)
                    timeFactor_expanded = timeFactor.reshape(*([1] * source_mat.ndim), *target_mat.shape)

                    wheight_mat += reward * timeFactor_expanded * learning_rate * ( ( source_expanded - wheight_mat ) * target_expanded )
                
                    return output, output_rev, wheight_mat
                euler_func = eulerStep
        else:
            raise ValueError("Invalid learning rule specified. Must be 'instar' or 'outstar'.")

    return euler_func, reward_func
        


class HebbianConnection(Step):

    def __init__(self, name, params):
        mandatory_params = ["shape", "target_shape", "tau", "tau_decay", "learning_rate", "learning_rule", "bidirectional", "reward_duration"]

        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self._params["wheight_shape"] = self._params["shape"] + self._params["target_shape"]
        self._params["scalar_shape"] = (1,)

        self._euler_func, self.reward_func = generate_euler_func(util_jax.cfg['euler_step_static_precompile'], self._params)

        self.needs_input_connections = True
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = 3
        self._delta_t = util_jax.get_config()["delta_t"]

        #self.register_input(util.DEFAULT_INPUT_SLOT) # source activation
        self.register_input("in1") # target activaiton
        self.register_input("in2") # reward signal
        self.register_output("out1") # rev_output

        self.register_buffer("wheights", "wheight_shape") # dynamic wheight parameter
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
        reward, onset, timer = self.reward_func(self._delta_t, reward_signal, self.buffer["reward_onset"], self.buffer["reward_timer"], self._params["reward_duration"])
        output, output_rev, wheights = self._euler_func(self._delta_t, prng_key, self.buffer["wheights"], source_mat, target_mat, reward, self._params["learning_rate"], self._params["tau"], self._params["tau_decay"])

        # Return output and buffer update
        return {util.DEFAULT_OUTPUT_SLOT: output, "out1": output_rev, 
                "wheights": wheights, "reward_timer": timer, "reward_onset": onset}
    
    def reset(self): #Override
        self.buffer["wheights"] = util_jax.zeros(self._params["shape"]+self._params["target_shape"])
        self.buffer["reward_timer"] = util_jax.zeros((1,))
        self.buffer["reward_onset"] = util_jax.zeros((1,))
        self.reset_buffer(util.DEFAULT_OUTPUT_SLOT, slot_shape="target_shape")
        self.reset_buffer("out2", slot_shape="shape")
