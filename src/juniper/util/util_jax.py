from ..core.backend.Exceptions import JuniperConfigurationError
import os
from . import util
import json
import numpy as np
import jax.numpy as jnp
import jax

def _load_config():
    cfg = json.load(open(os.path.join(util.root(), "run_config.json"), "r"))

    # Parse dtype
    unsupported_dtypes = ["float64"]
    if "dtype" in cfg:
        dtype_str = cfg["dtype"]
        if cfg["dtype"] in unsupported_dtypes:
            raise JuniperConfigurationError(f"dtype {cfg['dtype']} not supported")
        cfg["dtype"] = np.dtype(dtype_str).type
        cfg["jdtype"] = jnp.dtype(dtype_str).type
    return cfg

_cfg_instance = None

def get_config():
    global _cfg_instance
    if _cfg_instance is None:
        _cfg_instance = _load_config()
    return _cfg_instance

cfg = get_config()

if cfg['debug']:
    jax_prng_key = jax.random.key(42)
    np.random.seed(42)
else:
    jax_prng_key = jax.random.key(np.random.randint(0, 2**16))

def next_random_key():
    global jax_prng_key
    jax_prng_key, subkey = jax.random.split(jax_prng_key)
    return subkey
def next_random_keys(num):
    global jax_prng_key
    keys = jax.random.split(jax_prng_key, num+1)
    jax_prng_key = keys[0]
    return keys[1:]

def build_prng_tree(kernel_map, dynamic_paths, static_key):
    tree = {}
    slots = []
    for element_path in kernel_map.keys():
        tree[element_path] = static_key
        if element_path in dynamic_paths:
            slots.append((tree, element_path))
    return tree, slots

def update_prng_tree(prng_tree, prng_slots):
    if len(prng_slots) == 0:
        return prng_tree
    keys = next_random_keys(len(prng_slots))
    for key, (tree, element_path) in zip(keys, prng_slots):
        tree[element_path] = key
    return prng_tree

def zeros(shape, dtype=None):
    return jnp.zeros(shape, dtype=dtype or cfg["jdtype"])

def ones(shape, dtype=None):
    return jnp.ones(shape, dtype=dtype or cfg["jdtype"])

def constant(shape, dtype, value):
    return ones(shape, dtype=dtype) * value

def dtype_CV_string():
    dtype = cfg["dtype"]
    if dtype == np.float32:
        return "CV_32F"
    elif dtype == np.float16:
        return "CV_16F"
    else:
        return "CV_undefined"
