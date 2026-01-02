from ..configurables.Step import Step
from ..util import util, util_jax

import numpy as np
import jax.numpy as jnp
from jax import lax


def _argmax_center_jax(activation: jnp.ndarray):
    """
    Returns (cx, cy, peak) all as JAX scalars (no Python int/float conversion).
    cx, cy are normalized to [0, 1).
    """
    h, w = activation.shape
    flat_idx = jnp.argmax(activation)              # scalar int
    peak = jnp.max(activation)                     # scalar

    y = flat_idx // w
    x = flat_idx - y * w

    cx = x.astype(jnp.float32) / jnp.float32(w)
    cy = y.astype(jnp.float32) / jnp.float32(h)

    eps = jnp.asarray(1e-12, dtype=cx.dtype)
    cx = jnp.clip(cx, 0.0, 1.0 - eps)
    cy = jnp.clip(cy, 0.0, 1.0 - eps)
    return cx, cy, peak


def _pad_and_crop_jax(img: jnp.ndarray, left: jnp.ndarray, top: jnp.ndarray, vw: int, vh: int):
    """
    JAX-friendly pad + dynamic crop.
    left/top can be traced scalars (int32).
    vw/vh must be Python ints (static, from params).
    """
    hvw = int(round(vw / 2))
    hvh = int(round(vh / 2))

    padded = jnp.pad(img, ((hvh, hvh), (hvw, hvw), (0, 0)), mode="constant", constant_values=0)

    # shift into padded coordinates
    left_p = left + jnp.int32(hvw)
    top_p  = top  + jnp.int32(hvh)

    # clamp so dynamic_slice stays in-bounds
    H2, W2, _ = padded.shape
    max_left = jnp.int32(W2 - vw)
    max_top  = jnp.int32(H2 - vh)
    left_p = jnp.clip(left_p, 0, max_left)
    top_p  = jnp.clip(top_p,  0, max_top)

    return lax.dynamic_slice(padded, (top_p, left_p, 0), (vh, vw, padded.shape[2]))


def _set_one_hot(kernel: jnp.ndarray, cy: jnp.ndarray, cx: jnp.ndarray):
    """
    Places a 1.0 at the nearest (ky, kx) location.
    """
    print(kernel.shape)
    kh, kw = kernel.shape
    ky = jnp.clip(jnp.round(cy * kh).astype(jnp.int32), 0, kh - 1)
    kx = jnp.clip(jnp.round(cx * kw).astype(jnp.int32), 0, kw - 1)
    return jnp.zeros_like(kernel).at[ky, kx].set(1.0)


# ---------- main factory ----------

def compute_kernel_factory(params, delta_t):
    # static (Python) values captured in closure
    H, W = params["input_shape"]
    vh, vw = params["viewport_size"]
    learn_interval = params["learn_interval_s"]
    learn_total = params["learn_total_s"]
    saccade_duration = params["saccade_duration_s"]
    move_eps = params["move_eps"]
    threshold = params["threshold"]
    simplified = params["simplified"]
    kernel_shape = tuple(params["kernel_shape"])  # ensure Python tuple for shape checks

    dt = float(delta_t)  # static scalar (or keep as jnp.asarray if you prefer)

    def compute_kernel(input_mats, buffer, **kwargs):
        img = input_mats[util.DEFAULT_INPUT_SLOT]               # (H,W,3) jax array
        activation = input_mats["viewport_center"]              # (kh,kw) jax array
        learn_in = input_mats["learn_mode"]                     # jax array (scalar or small tensor)

        CoS = buffer["CoS"]                                     # typically shape (1,) or ()
        kernel = buffer["kernel"]                               # (kh,kw)
        lastX = buffer["lastX"]                                 # scalar
        lastY = buffer["lastY"]                                 # scalar
        startSC = buffer["startSC"]                             # scalar (0/1 or bool)
        endSC = buffer["endSC"]                                 # scalar (0/1 or bool)
        elapsed_time = buffer["elapsed_time"]                   # scalar
        elapsed_learn_time = buffer["elapsed_learn_time"]       # scalar

        # Make sure these are scalar booleans (works whether they came as 0/1 or bool)
        startSC_b = jnp.asarray(startSC > 0)
        endSC_b   = jnp.asarray(endSC > 0)

        # Early return if viewport bigger than image (static check)
        if vw > W or vh > H:
            out0 = jnp.zeros((vh, vw, 3), dtype=img.dtype)
            return {
                util.DEFAULT_OUTPUT_SLOT: out0,
                "kernel": kernel,
                "CoS": CoS,
                "startSC": startSC,
                "endSC": endSC,
                "elapsed_time": elapsed_time,
                "elapsed_learn_time": elapsed_learn_time,
                "lastX": lastX,
                "lastY": lastY,
            }
    # test rest of function

        # If kernel shape mismatches activation shape (static decision), reset state
        if kernel_shape != tuple(activation.shape):
            kernel = jnp.zeros_like(activation)
            lastX = jnp.asarray(0.5, dtype=jnp.float32)
            lastY = jnp.asarray(0.5, dtype=jnp.float32)
            startSC_b = jnp.asarray(False)
            endSC_b = jnp.asarray(False)
            elapsed_time = jnp.asarray(0.0, dtype=jnp.float32)

        def no_activation_branch(_):
            out = jnp.zeros((vh, vw, 3), dtype=img.dtype)
            kernel0 = jnp.zeros_like(activation)
            CoS0 = jnp.zeros_like(CoS)
            return {
                util.DEFAULT_OUTPUT_SLOT: out,
                "kernel": kernel0,
                "CoS": CoS0,
                "startSC": jnp.asarray(0, dtype=startSC.dtype) if hasattr(startSC, "dtype") else jnp.asarray(0),
                "endSC": jnp.asarray(0, dtype=endSC.dtype) if hasattr(endSC, "dtype") else jnp.asarray(0),
                "elapsed_time": elapsed_time * 0,
                "elapsed_learn_time": elapsed_learn_time * 0,
                "lastX": jnp.asarray(0.5, dtype=jnp.float32),
                "lastY": jnp.asarray(0.5, dtype=jnp.float32),
            }

        def has_activation_branch(_):
            cx, cy, peak = _argmax_center_jax(activation)

            # ---------- simplified path ----------
            def simplified_branch(_):
                def below_thresh(_):
                    out = jnp.zeros((vh, vw, 3), dtype=img.dtype)
                    return out, jnp.zeros_like(kernel), jnp.zeros_like(CoS), startSC_b, endSC_b, elapsed_time, elapsed_learn_time, lastX, lastY

                def above_thresh(_):
                    k1 = _set_one_hot(kernel, cy, cx)

                    left = jnp.round(cx * W).astype(jnp.int32)
                    top  = jnp.round(cy * H).astype(jnp.int32)
                    out = _pad_and_crop_jax(img, left, top, vw, vh)
                    return out, k1, jnp.zeros_like(CoS), startSC_b, endSC_b, elapsed_time, elapsed_learn_time, lastX, lastY

                out, k1, cos1, s1, e1, t1, lt1, lx1, ly1 = lax.cond(
                    peak < jnp.asarray(threshold, dtype=peak.dtype),
                    below_thresh,
                    above_thresh,
                    operand=None
                )

                return {
                    util.DEFAULT_OUTPUT_SLOT: out,
                    "kernel": k1,
                    "CoS": cos1,
                    "startSC": s1.astype(startSC.dtype) if hasattr(startSC, "dtype") else s1,
                    "endSC": e1.astype(endSC.dtype) if hasattr(endSC, "dtype") else e1,
                    "elapsed_time": t1,
                    "elapsed_learn_time": lt1,
                    "lastX": lx1,
                    "lastY": ly1,
                }

            # ---------- full path ----------
            def full_branch(_):
                moved = (jnp.abs(lastX - cx) > move_eps) | (jnp.abs(lastY - cy) > move_eps)
                peak_ok = peak > jnp.asarray(threshold, dtype=peak.dtype)
                can_start = (~startSC_b) & (~endSC_b) & moved & peak_ok

                # start a saccade: update lastX/lastY + startSC + kernel one-hot + reset elapsed_time
                def start_saccade(_):
                    k1 = _set_one_hot(kernel, cy, cx)
                    return (cx, cy, jnp.asarray(True), jnp.asarray(False), jnp.asarray(0.0, dtype=elapsed_time.dtype), k1)

                def no_start(_):
                    return (lastX, lastY, startSC_b, endSC_b, elapsed_time, kernel)

                lastX1, lastY1, startSC1, endSC1, elapsed_time1, kernel1 = lax.cond(
                    can_start,
                    start_saccade,
                    no_start,
                    operand=None
                )

                # during startSC: output zeros; after duration, flip to endSC and set CoS=1 briefly
                def startSC_branch(_):
                    out = jnp.zeros((vh, vw, 3), dtype=img.dtype)

                    def finish(_):
                        return (jnp.asarray(False), jnp.asarray(True), jnp.asarray(0.0, dtype=elapsed_time1.dtype), jnp.ones_like(CoS), out)

                    def cont(_):
                        return (jnp.asarray(True), jnp.asarray(False), elapsed_time1 + dt, jnp.zeros_like(CoS), out)

                    return lax.cond(elapsed_time1 > saccade_duration, finish, cont, operand=None)

                def not_startSC_branch(_):
                    # maybe still in endSC
                    def endSC_update(_):
                        def end_finish(_):
                            return (jnp.asarray(False), jnp.asarray(0.0, dtype=elapsed_time1.dtype), jnp.zeros_like(CoS))
                        def end_cont(_):
                            return (jnp.asarray(True), elapsed_time1 + dt, CoS)
                        return lax.cond(elapsed_time1 > saccade_duration, end_finish, end_cont, operand=None)

                    endSC2, elapsed_time2, CoS2 = lax.cond(endSC1, endSC_update, lambda _: (endSC1, elapsed_time1, CoS), operand=None)

                    # compute viewport crop around last fixation
                    left = jnp.round(lastX1 * W).astype(jnp.int32)
                    top  = jnp.round(lastY1 * H).astype(jnp.int32)

                    # learning-mode jitter
                    learn_on = jnp.max(learn_in) > 0.5

                    hvw = int(round(vw / 2))
                    hvh = int(round(vh / 2))

                    def learn_adjust(_):
                        lt = elapsed_learn_time + dt

                        # only apply jitter if learn_interval <= lt < learn_total
                        in_window = (lt >= learn_interval) & (lt < learn_total)

                        step = jnp.floor((lt - learn_interval) / learn_interval).astype(jnp.int32)

                        # top adjustments by step ranges
                        top_adj = jnp.where((step >= 0) & (step <= 2),  hvh // 2, 0)
                        top_adj = top_adj + jnp.where((step >= 3) & (step <= 5),  hvh // 4, 0)
                        top_adj = top_adj + jnp.where((step >= 9) & (step <= 11), -hvh // 4, 0)
                        top_adj = top_adj + jnp.where((step >= 12) & (step <= 14), -hvh // 2, 0)

                        # left adjustments by step set membership
                        left_adj = jnp.where((step == 0) | (step == 3) | (step == 6) | (step == 9) | (step == 12),  hvw // 2, 0)
                        left_adj = left_adj + jnp.where((step == 2) | (step == 5) | (step == 8) | (step == 11) | (step == 14), -hvw // 2, 0)

                        left2 = left + jnp.where(in_window, jnp.int32(left_adj), jnp.int32(0))
                        top2  = top  + jnp.where(in_window, jnp.int32(top_adj),  jnp.int32(0))
                        return left2, top2, lt

                    def no_learn_adjust(_):
                        return left, top, elapsed_learn_time * 0

                    left2, top2, lt2 = lax.cond(learn_on, learn_adjust, no_learn_adjust, operand=None)

                    out = _pad_and_crop_jax(img, left2, top2, vw, vh)

                    return (out, endSC2, elapsed_time2, lt2, CoS2)

                # choose between startSC branch and not-startSC branch
                start_out = startSC_branch(None)
                not_start_out = not_startSC_branch(None)

                out, startSC2, endSC2, elapsed_time2, CoS2 = lax.cond(
                    startSC1,
                    lambda _: start_out,  # unpacked tuple
                    lambda _: (not_start_out[0], jnp.asarray(False), not_start_out[1], not_start_out[2], not_start_out[4]),
                    operand=None
                )

                # For not_startSC path, startSC is False; already handled above.
                # For startSC path, elapsed_learn_time remains unchanged.
                elapsed_learn_time2 = lax.cond(
                    startSC1,
                    lambda _: elapsed_learn_time,
                    lambda _: not_start_out[3],
                    operand=None
                )

                return {
                    util.DEFAULT_OUTPUT_SLOT: out,
                    "kernel": kernel1,
                    "CoS": CoS2,
                    "startSC": startSC2.astype(startSC.dtype) if hasattr(startSC, "dtype") else startSC2,
                    "endSC": endSC2.astype(endSC.dtype) if hasattr(endSC, "dtype") else endSC2,
                    "elapsed_time": elapsed_time2,
                    "elapsed_learn_time": elapsed_learn_time2,
                    "lastX": lastX1,
                    "lastY": lastY1,
                }

            return lax.cond(
                jnp.asarray(simplified),
                simplified_branch,
                full_branch,
                operand=None
            )

        return lax.cond(jnp.max(activation) > 0, has_activation_branch, no_activation_branch, operand=None)

    return compute_kernel


class ViewportCamera(Step):
 
    def __init__(self, name, params):
        mandatory_params = ["input_shape", "kernel_shape", "viewport_size"]
        params["shape"] = params["viewport_size"] + (3,)
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self._params["CoS_shape"] = (1,)
        self._params["output_shape"] = self._params["viewport_size"] + (3,)
        self._delta_t = float(util_jax.get_config()["delta_t"])

        if "simplified" not in self._params.keys():
            self._params["simplified"] = False
        if "threshold"  not in self._params.keys():
            self._params["threshold"] = 0.5
        if "move_eps" not in self._params.keys():
            self._params["move_eps"] = 0.05
        if "saccade_duration_s"  not in self._params.keys():
            self._params["saccade_duration_s"] = 0.020
        if "learn_interval_s" not in self._params.keys():
            self._params["learn_interval_s"] = 0.025
        if "learn_total_s"  not in self._params.keys():
            self._params["learn_total_s"] = 0.325

        self.register_input("viewport_center")
        self.register_input("learn_mode")
        self.register_output("kernel")
        self.register_output("CoS")
        
        self.register_buffer("startSC")
        self.register_buffer("endSC")
        self.register_buffer("elapsed_time")
        self.register_buffer("elapsed_learn_time")
        self.register_buffer("lastX")
        self.register_buffer("lastY")

        self.compute_kernel = compute_kernel_factory(self._params, self._delta_t)

        self.reset()

    def reset(self): # Override default reset, to handle shapes of buffer explicitly.
        self.buffer["startSC"] = jnp.int32(0) #util_jax.zeros((1,))
        self.buffer["endSC"] = jnp.int32(0)
        self.buffer["elapsed_time"] = jnp.float32(0)
        self.buffer["elapsed_learn_time"] =  jnp.float32(0)
        self.buffer["lastX"] = jnp.float32(0)
        self.buffer["lastY"] = jnp.float32(0)
        self.reset_buffer(util.DEFAULT_OUTPUT_SLOT, slot_shape="output_shape")
        self.reset_buffer("kernel", slot_shape="kernel_shape")
        self.reset_buffer("CoS", slot_shape="CoS_shape")
        reset_state = {}
        reset_state["startSC"] = self.buffer["startSC"]
        reset_state["endSC"] = self.buffer["endSC"]
        reset_state["elapsed_time"] = self.buffer["elapsed_time"]
        reset_state["elapsed_learn_time"] = self.buffer["elapsed_learn_time"]
        reset_state["lastX"] = self.buffer["lastX"]
        reset_state["lastY"] = self.buffer["lastY"]
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self.buffer[util.DEFAULT_OUTPUT_SLOT]
        reset_state["kernel"] = self.buffer["kernel"]
        reset_state["CoS"] = self.buffer["CoS"]
        return reset_state

    
    
