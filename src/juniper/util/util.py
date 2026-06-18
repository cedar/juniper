from ..core.backend.Exceptions import JuniperConfigurationError
import os
import numpy as np
import time
import atexit
import functools
ROOT_FOLDER = ""
DEFAULT_INPUT_SLOT = "in0"
DEFAULT_OUTPUT_SLOT = "out0"

ROOT_FOLDER = os.path.dirname(__file__)

if not os.path.exists(os.path.join(ROOT_FOLDER, "run_config.json")):
    raise JuniperConfigurationError(f"No run_config found in root folder {os.path.join(ROOT_FOLDER )}. Please specify the correct path in the first line of util.py")

def root():
    return ROOT_FOLDER

def prettify(num_2_prettify):
    return f'{np.mean(num_2_prettify):.5f} +- {np.std(num_2_prettify):.5f}  (min: {np.min(num_2_prettify)})'

## --- Timed print ---
_last_time = time.time()
_overall_start_time = _last_time

def tprint(label=None):
    global _last_time
    current_time = time.time()
    print_string = f"{current_time - _last_time:>7.3f}s -"
    if label is not None:
        print_string += f" {label} -"
    print(print_string)
    _last_time = current_time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func_return = func(*args, **kwargs)
        end = time.perf_counter()
        return end-start, func_return
    return wrapper

def _exit_handler():
    print(f"# {time.time() - _overall_start_time:>7.3f}s - Total runtime #")

def _is_scalar_shape(shape):
    return shape == () or shape == (1,)

atexit.register(_exit_handler)
