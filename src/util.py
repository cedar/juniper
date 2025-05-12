ROOT_FOLDER = ""
DEFAULT_INPUT_SLOT = "in0"
DEFAULT_OUTPUT_SLOT = "out0"

import os
import numpy as np
import time
import atexit

if not os.path.exists(os.path.join(ROOT_FOLDER, "run_config.json")):
    raise Exception(f"No run_config found in root folder {ROOT_FOLDER}. Please specify the correct path in the first line of util.py")

def root():
    return ROOT_FOLDER

def prettify(l):
    return f'{np.mean(l):.5f} +- {np.std(l):.5f}  (min: {np.min(l)})'

_architecture = None
def set_architecture(arch):
    global _architecture
    if _architecture is None:
        _architecture = arch
    else:
        raise Exception("Architecture already set. Did you create two architectures?")

def get_architecture():
    if _architecture is None:
        raise Exception("No architecture was created yet.")
    return _architecture

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
    if not label is None:
        print_string += f" {label} -"
    print(print_string)
    _last_time = current_time

def _exit_handler():
    print(f"# {time.time() - _overall_start_time:>7.3f}s - Total runtime #")

atexit.register(_exit_handler)