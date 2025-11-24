# juniper_fast_prebatched.py
# =============================================================
# JUNIPER FAST - Prebatched, JITted engine for 1D NeuralFields
# Subset supported: NeuralField (dynamic), GaussInput (static source), StaticGain (static multiplier)
# Option A: field shapes are fixed at architecture build time.
# =============================================================
import time
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

# -------------------------
# Steps
# -------------------------
class Step:
    def __init__(self, name):
        self.name = name
        self.inputs = []  # names of upstream steps
        self.consumers = []  # names of downstream steps

    def get_name(self):
        return self.name

class GaussInput(Step):
    def __init__(self, name, shape, center, amplitude, width):
        super().__init__(name)
        self.shape = shape  # tuple like (N,)
        self._center = float(center)
        self._amplitude = float(amplitude)
        self._width = float(width)

    def export_array(self):
        N = int(self.shape[0])
        x = jnp.arange(N)
        g = self._amplitude * jnp.exp(-0.5 * ((x - self._center) / self._width) ** 2)
        return g  # 1D jax array

class StaticGain(Step):
    def __init__(self, name, gain):
        super().__init__(name)
        self.gain = float(gain)

    def export_scalar(self):
        return float(self.gain)

class NeuralField(Step):
    def __init__(self, name, shape, kernel, tau=10.0, beta=1.0, theta=0.0, rest=0.0, inhib=0.0, noise=0.0):
        super().__init__(name)
        self.shape = shape  # tuple (N,)
        self.kernel = jnp.asarray(kernel)
        self.tau = float(tau)
        self.beta = float(beta)
        self.theta = float(theta)
        self.rest = float(rest)
        self.inhib = float(inhib)
        self.noise = float(noise)
        # initial state (can be nonzero if desired)
        self.u0 = jnp.zeros(shape)

# -------------------------
# Engine / Compiler
# -------------------------
@dataclass
class EngineState:
    u_batch: jnp.ndarray  # (B, Lmax)

@dataclass
class EngineParams:
    kernel_batch: jnp.ndarray  # (B, Kmax)
    mask: jnp.ndarray  # (B, Lmax)
    tau: jnp.ndarray  # (B,)
    beta: jnp.ndarray  # (B,)
    theta: jnp.ndarray  # (B,)
    rest: jnp.ndarray  # (B,)
    inhib: jnp.ndarray  # (B,)
    noise: jnp.ndarray  # (B,)
    I_static: jnp.ndarray  # (B, Lmax) precomputed static contribution

@dataclass
class JaxEngine:
    step_jit: any
    state: EngineState
    params: EngineParams

    def run_steps(self, n_steps: int, key: jax.Array):
        def body(carry, _):
            state, key = carry
            key, sub = jax.random.split(key)
            state = self.step_jit(state, self.params, dt, sub)
            return (state, key), None
        dt = 1.0  # default dt used; step_jit expects dt as an arg (we embed dt below)
        (final_state, _), _ = jax.lax.scan(body, (self.state, key), None, length=n_steps)
        self.state = final_state
        return final_state

# -------------------------
# Helper: pad lists into batch arrays (used only at compile time)
# -------------------------
def pad_list_to_batch(list_of_1d):
    if len(list_of_1d) == 0:
        return jnp.zeros((0, 0)), jnp.zeros((0, 0))
    lengths = [int(a.shape[0]) for a in list_of_1d]
    Lmax = max(lengths)
    padded = [a if a.shape[0] == Lmax else jnp.pad(a, (0, Lmax - a.shape[0])) for a in list_of_1d]
    batch = jnp.stack(padded, axis=0)
    mask = (jnp.arange(Lmax)[None, :] < jnp.array(lengths)[:, None]).astype(jnp.float32)
    return batch, mask, lengths

from jax import tree_util

tree_util.register_pytree_node(
    EngineState,
    lambda s: ((s.u_batch,), None),
    lambda _, xs: EngineState(xs[0])
)

tree_util.register_pytree_node(
    EngineParams,
    lambda p: ((p.kernel_batch, p.mask, p.tau, p.beta, p.theta, p.rest, p.inhib, p.noise, p.I_static), None),
    lambda _, xs: EngineParams(*xs)
)

# -------------------------
# Fast batched NF Euler update (JIT, GPU-friendly)
# -------------------------
@partial(jax.jit, static_argnames=[])
def batched_nf_step(u_batch, I_batch, kernel_batch, rest_arr, inhib_arr, tau_arr, beta_arr, theta_arr, noise_arr, mask, dt, prng_keys):
    # u_batch: (B, L)
    # I_batch: (B, L)
    # kernel_batch: (B, K)
    B, L = u_batch.shape
    _, K = kernel_batch.shape

    # activation
    beta_b = beta_arr[:, None]
    theta_b = theta_arr[:, None]
    f_u = 1.0 / (1.0 + jnp.exp(-beta_b * (u_batch - theta_b)))  # (B, L)

    # conv per-batch via lax.conv_general_dilated
    f_u_nwc = f_u[..., None]  # (B, L, 1)

    def conv_single(f_row, k_row):
        x = f_row[None, :, :]  # (1, L, 1)
        k = k_row[:, None, None]  # (K, 1, 1)
        y = lax.conv_general_dilated(x, k, window_strides=(1,), padding="SAME",
                                     dimension_numbers=("NWC", "WIO", "NWC"))
        return y[0, :, 0]

    conv = jax.vmap(conv_single, in_axes=(0,0))(f_u_nwc, kernel_batch)  # (B, L)

    # global inhibition scalar per field
    G = jnp.sum(f_u * mask, axis=1)[:, None]  # (B,1)

    # du
    du = (-u_batch + conv + rest_arr[:, None] + I_batch - inhib_arr[:, None] * G) / tau_arr[:, None]

    # noise
    normals = jax.vmap(lambda k: jax.random.normal(k, (L,)))(prng_keys)
    u_next = u_batch + dt * du + noise_arr[:, None] * normals

    # zero out padded region
    u_next = u_next * mask

    f_next = 1.0 / (1.0 + jnp.exp(-beta_b * (u_next - theta_b)))
    return u_next, f_next

# -------------------------
# Compiler: build batched arrays and the single jitted step function
# -------------------------
def compile_architecture(steps_dict, connections):
    """
    steps_dict: name -> Step object
    connections: name -> list of consumer names
    """
    # 1) collect NFs and their static providers
    nfs = []
    nf_names = []
    for name, s in steps_dict.items():
        if isinstance(s, NeuralField):
            nfs.append(s)
            nf_names.append(name)

    if len(nfs) == 0:
        raise RuntimeError("No NeuralFields found")

    B = len(nfs)
    lengths = [int(nf.shape[0]) for nf in nfs]
    Lmax = max(lengths)

    # initial u batch and mask
    u0_list = [nf.u0 for nf in nfs]
    u_batch, mask, _ = pad_list_to_batch(u0_list)

    # per-field scalar params
    tau_arr = jnp.asarray([nf.tau for nf in nfs])
    beta_arr = jnp.asarray([nf.beta for nf in nfs])
    theta_arr = jnp.asarray([nf.theta for nf in nfs])
    rest_arr = jnp.asarray([nf.rest for nf in nfs])
    inhib_arr = jnp.asarray([nf.inhib for nf in nfs])
    noise_arr = jnp.asarray([nf.noise for nf in nfs])

    # kernels into (B, Kmax)
    kernel_list = [nf.kernel for nf in nfs]
    kernel_batch, _, _ = pad_list_to_batch(kernel_list)

    # 2) STATIC GRAPH: since GaussInput and StaticGain are constant in this subset,
    # we compute the *total static contribution* to each NF's input at compile time.
    # For each NF, sum the outputs of all static providers connected to it, with gains applied.
    # We only support static sources that are either GaussInput OR StaticGain chaining a GaussInput.
    # Build mapping: for each NF name, find its list of upstream static steps (depth 1 or 2)
    static_I_list = []
    for nf_name in nf_names:
        nf_obj = steps_dict[nf_name]
        incoming = nf_obj.inputs  # names
        # accumulate an array sized (nf_length,)
        total = jnp.zeros((int(nf_obj.shape[0]),))
        if len(incoming) == 0:
            total = total  # empty
        else:
            # for each incoming static source name, resolve value
            for src in incoming:
                src_obj = steps_dict[src]
                if isinstance(src_obj, GaussInput):
                    g = src_obj.export_array()
                    # if shapes differ, pad/truncate to nf length
                    L = int(nf_obj.shape[0])
                    if g.shape[0] < L:
                        g = jnp.pad(g, (0, L - g.shape[0]))
                    elif g.shape[0] > L:
                        g = g[:L]
                    total = total + g
                elif isinstance(src_obj, StaticGain):
                    # StaticGain must have its own upstream (GaussInput)
                    # expect exactly one input
                    if len(src_obj.inputs) != 1:
                        raise RuntimeError("StaticGain must have exactly one upstream GaussInput for this simplified engine")
                    upstream = steps_dict[src_obj.inputs[0]]
                    if not isinstance(upstream, GaussInput):
                        raise RuntimeError("StaticGain upstream must be GaussInput in this simplified engine")
                    g = upstream.export_array()
                    L = int(nf_obj.shape[0])
                    if g.shape[0] < L:
                        g = jnp.pad(g, (0, L - g.shape[0]))
                    elif g.shape[0] > L:
                        g = g[:L]
                    total = total + src_obj.export_scalar() * g
                else:
                    raise RuntimeError("Unsupported static upstream type in this simplified compile path")
        static_I_list.append(total)

    # pad static_I_list to (B, Lmax)
    I_static_batch, _, _ = pad_list_to_batch(static_I_list)

    # 3) Build EngineState and EngineParams
    state = EngineState(u_batch=u_batch)
    params = EngineParams(
        kernel_batch=kernel_batch,
        mask=mask,
        tau=tau_arr,
        beta=beta_arr,
        theta=theta_arr,
        rest=rest_arr,
        inhib=inhib_arr,
        noise=noise_arr,
        I_static=I_static_batch
    )

    # 4) Build step_jit function: it simply uses I_static + possibly additional dynamic inputs
    #    For the small subset we support, the total input I_batch == I_static (no per-tick variable inputs)
    def step_fn(state: EngineState, params: EngineParams, dt: float, prng_key):
        # Use precomputed static input
        I_batch = params.I_static  # (B, Lmax)
        u_batch = state.u_batch
        # prepare prng keys: generate one per field
        keys = jax.random.split(prng_key, u_batch.shape[0])
        u_next, f_next = batched_nf_step(
            u_batch, I_batch,
            params.kernel_batch,
            params.rest, params.inhib,
            params.tau, params.beta, params.theta,
            params.noise, params.mask, dt, keys
        )
        # return new state (EngineState)
        return EngineState(u_batch=u_next)

    step_jit = jax.jit(step_fn)
    engine = JaxEngine(step_jit=step_jit, state=state, params=params)
    return engine

# -------------------------
# Demo & benchmark (main)
# -------------------------
if __name__ == "__main__":
    import numpy as onp
    print("Building architecture (100 heterogeneous 1D fields)...")
    rng = onp.random.default_rng(0)
    arch_steps = {}
    connections = {}

    Nfields = 100
    # Create fields and static Gaussian sources
    for i in range(Nfields):
        N = int(rng.integers(30, 200))
        # create kernel (small)
        k_len = 25
        centers = jnp.linspace(-3, 3, k_len)
        k = jnp.exp(-centers**2)
        nf = NeuralField(f"f{i}", shape=(N,), kernel=k, tau=10.0 + rng.uniform(-2,2), beta=1.5, theta=0.0, rest=-0.3, inhib=0.001, noise=0.01)
        arch_steps[nf.name] = nf
        connections[nf.name] = []

        # Gaussian source
        g = GaussInput(f"g{i}", shape=(N,), center=N/2, amplitude=1.0, width=max(1.0, N/10))
        arch_steps[g.name] = g
        connections[g.name] = [nf.name]  # g -> nf
        nf.inputs.append(g.name)

    # compile
    print("Compiling engine (precomputing batched arrays)...")
    t0c = time.time()
    engine = compile_architecture(arch_steps, connections)
    t1c = time.time()
    print(f"Compile time: {t1c - t0c:.3f} s (includes JIT compilation on first tick)")

    # warm-up (first call triggers JIT)
    print("Warmup step (this will JIT compile the kernels)...")
    key = jax.random.PRNGKey(42)
    dt = 1.0
    t0 = time.time()
    engine.state = engine.step_jit(engine.state, engine.params, dt, key)  # JIT compile & run
    engine.state.u_batch.block_until_ready()
    t1 = time.time()
    print(f"Warmup took {t1 - t0:.4f} s (includes compile)")

    # benchmark multiple ticks
    n_ticks = 200
    total_times = []
    kernel_times = []
    tstart = time.time()
    key = jax.random.PRNGKey(0)
    for i in range(n_ticks):
        key, sub = jax.random.split(key)
        t0 = time.time()
        engine.state = engine.step_jit(engine.state, engine.params, dt, sub)
        engine.state.u_batch.block_until_ready()
        t1 = time.time()
        total_times.append(t1 - t0)
    tend = time.time()

    avg_total_ms = (sum(total_times) / len(total_times)) * 1000.0
    print("=======================================")
    print(f"Ticks: {n_ticks}, Fields: {Nfields}")
    print(f"Avg total tick time (ms): {avg_total_ms:0.3f} ms")
    print(f"Total elapsed (s): {tend - tstart:0.3f} s")
    print("=======================================")
