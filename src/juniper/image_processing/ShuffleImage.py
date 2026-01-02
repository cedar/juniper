from ..configurables.Step import Step
from ..util import util, util_jax

import jax.numpy as jnp
import jax
import numpy as np

def compute_kernel_factory(params, delta_t):
    # static (Python) values captured in closure
    H, W = params["input_shape"]
    vh, vw = params["viewport_size"]
    H0 = int(H/2)
    W0 = int(W/2)
    hvw = int(np.round(vw / 2))
    hvh = int(np.round(vh / 2))
    learn_interval = params["learn_interval_s"]
    learn_total = params["learn_total_s"]
    threshold = params["threshold"]
    dt = float(delta_t)  # static scalar (or keep as jnp.asarray if you prefer)

    def learn_timer(learn_node, learn_onset, elapsed_learn_time):
        cond1 = learn_onset == 1
        cond2 = jnp.logical_and(learn_onset == 0, learn_node > threshold)
        cond3 = jnp.logical_and(learn_onset == 0, learn_node < threshold)

        learn_onset = jnp.select(
            condlist=[cond1, cond2, cond3],
            choicelist=[1 - ( elapsed_learn_time >= (learn_total) ), jnp.float32(1), jnp.float32(0)]
        )
        elapsed_learn_time = jnp.select(
            condlist=[cond1, cond2, cond3],
            choicelist=[elapsed_learn_time, jnp.float32(0), jnp.float32(0)]
        )

        learn_active = learn_onset * (elapsed_learn_time < learn_total)

        return learn_active, learn_onset, elapsed_learn_time + dt
    
    def determine_slices_and_crop(input, elapsed_learn_time):
        left = jnp.round(W0)
        top = jnp.round(H0)

        step = jnp.floor((elapsed_learn_time - learn_interval) / learn_interval)[0].astype(jnp.int32)

        top_delta = jnp.float32(0)

        top_delta = top_delta + jnp.where((step >= 0) & (step <= 2),  (hvh // 2), 0)
        top_delta = top_delta + jnp.where((step >= 3) & (step <= 5),  (hvh // 4), 0)
        top_delta = top_delta + jnp.where((step >= 9) & (step <= 11), -(hvh // 4), 0)
        top_delta = top_delta + jnp.where((step >= 12) & (step <= 14), -(hvh // 2), 0)
        top_delta = top_delta.astype(jnp.float32)

        top = top + top_delta

        plus_steps  = jnp.array([0, 3, 6, 9, 12], dtype=jnp.int32)
        minus_steps = jnp.array([2, 5, 8, 11, 14], dtype=jnp.int32)

        left_delta = jnp.float32(0)
        left_delta = left_delta + jnp.where(jnp.any(step == plus_steps),  hvw // 2, 0)
        left_delta = left_delta + jnp.where(jnp.any(step == minus_steps), -(hvw // 2), 0)
        
        left = left + left_delta

        top = jnp.round(top).astype(jnp.int32)
        left = jnp.round(left).astype(jnp.int32)

        padded = jnp.pad(input, ((hvh, hvh), (hvw, hvw), (0, 0)), mode="constant", constant_values=0)
        cropped = jax.lax.dynamic_slice(padded, (top,left,0), (vh,vw,3))

        return cropped

    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]
        learn_node = input_mats["learn_node"]
        learn_onset = buffer["learn_onset"]
        elapsed_learn_time = buffer["elapsed_learn_time"]

        learn_active, learn_onset, elapsed_learn_time = learn_timer(learn_node, learn_onset, elapsed_learn_time)

        cond1 = learn_active == 1
        cond2 = learn_active == 0

        output = jnp.select(
            condlist=[cond1, cond2],
            choicelist=[determine_slices_and_crop(input, elapsed_learn_time), input[H0-hvh:H0+hvh, W0-hvw:W0+hvw, :]]
        )
        
        return {util.DEFAULT_OUTPUT_SLOT: output,
                "elapsed_learn_time": elapsed_learn_time,
                "learn_onset": learn_onset}
    return compute_kernel


class ShuffleImage(Step):

    def __init__(self, name, params):
        mandatory_params = ["input_shape", "viewport_size"]
        params["shape"] = params["viewport_size"] + (3,)
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self._params["CoS_shape"] = (1,)
        self._params["output_shape"] = self._params["viewport_size"] + (3,)
        self._delta_t = float(util_jax.get_config()["delta_t"])

        if "threshold"  not in self._params.keys():
            self._params["threshold"] = 0.9
        if "learn_interval_s" not in self._params.keys():
            self._params["learn_interval_s"] = 0.025
        if "learn_total_s"  not in self._params.keys():
            self._params["learn_total_s"] = 0.325

        self.register_input("learn_node")
        
        self.register_buffer("elapsed_learn_time")
        self.register_buffer("learn_onset")

        self.compute_kernel = compute_kernel_factory(self._params, self._delta_t)
        self.reset()

    def reset(self): # Override default reset, to handle shapes of buffer explicitly.
        self.buffer["elapsed_learn_time"] =  jnp.float32(0)
        self.buffer["learn_onset"] = jnp.float32(0)
        self.reset_buffer(util.DEFAULT_OUTPUT_SLOT, slot_shape="output_shape")
        reset_state = {}
        reset_state["elapsed_learn_time"] = self.buffer["elapsed_learn_time"]
        reset_state["learn_onset"] = self.buffer["learn_onset"]
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self.buffer[util.DEFAULT_OUTPUT_SLOT]
        return reset_state