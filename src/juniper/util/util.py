import os
import numpy as np
import time
import atexit
ROOT_FOLDER = ""
DEFAULT_INPUT_SLOT = "in0"
DEFAULT_OUTPUT_SLOT = "out0"

ROOT_FOLDER = os.path.dirname(__file__)

if not os.path.exists(os.path.join(ROOT_FOLDER, "run_config.json")):
    raise Exception(f"No run_config found in root folder {os.path.join(ROOT_FOLDER )}. Please specify the correct path in the first line of util.py")

def root():
    return ROOT_FOLDER

def prettify(num_2_prettify):
    return f'{np.mean(num_2_prettify):.5f} +- {np.std(num_2_prettify):.5f}  (min: {np.min(num_2_prettify)})'

## --- Exceptions ---

class ArchitectureNotCompiledException(Exception):
    def __init__(self):
        super().__init__("Architecture needs to be compiled to perform this operation")


class ArchitectureCompiledException(Exception):
    def __init__(self):
        super().__init__("Architecture is already compiled, cannot perform this operation")

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

def _exit_handler():
    print(f"# {time.time() - _overall_start_time:>7.3f}s - Total runtime #")

atexit.register(_exit_handler)
