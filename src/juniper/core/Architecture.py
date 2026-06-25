from __future__ import annotations
import logging
import sys
from .frontend.Circuit import Circuit
from .frontend import CircuitContext
from .backend.Engine import Engine
from .backend.Engine import Recording
from .backend.Engine import TimingInfo
from .backend.Exceptions import CircuitError


logger = logging.getLogger(__name__)
_architecture_singleton = None

def get_arch(name : str = None) -> Architecture:
    global _architecture_singleton
    if _architecture_singleton is None:
        _architecture_singleton = Architecture() if name is None else Architecture(name=name)
    return _architecture_singleton

def delete_arch():
    global _architecture_singleton
    _architecture_singleton = None
    Circuit._current = None
    CircuitContext.set_current(None)

def init_logging(level : int = logging.INFO):
    """Initialize a default logging handler, which prints logs to console."""
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
            "%(levelname)s [%(name)s] %(message)s \n"
        )
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    logger.info("Setup default console logging.")

def init_logging_to_file(path : str, level : int = logging.INFO):
    file_handler = logging.FileHandler(filename=path, mode="a")
    formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s [%(name)s] %(message)s \n"
        )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logger.info("Setup default logging to file.")

class Architecture(Circuit):
    def __init__(self, name : str = "architecture"):
        """A singleton instance for the top-level circuit. Cannot have input or output slots.\n
        The architecture class also includes useful functions for comilation and simulation without manually having to call engine and compiler."""
        if Circuit._current is not None:
            raise CircuitError("A parent circuit already exists. Use Architecture class only to initialize the top-level architecture.")
        else:
            Circuit._current = self
            CircuitContext.set_current(self)
        super().__init__(name = name)
        self.engine = Engine()

    def set_arch_name(self, name : str):
        self._name = name

    def compile(self, warmup : int = 0, print_compile_info : bool = False, load_buffer : bool = False) -> None:
        self.engine.compile(circuit=self, warmup=warmup, print_compile_info=print_compile_info, load_buffer=load_buffer)

    def run_simulation(
            self,
            num_steps: int,
            steps_to_record: list[str] = [],
            print_timing: bool = True,
            save_buffer: bool = False,
        )-> tuple[Recording, TimingInfo]:
        return self.engine.run_simulation(num_steps=num_steps, steps_to_record=steps_to_record, print_timing=print_timing, save_buffer=save_buffer)

    def reset_state(self):
        self.engine.reset_state()

    def close_connections(self):
        self.engine._close_connections()

    def register_input_slot(self, input_slot_id, max_incoming_connections = 1):
        raise CircuitError("The top-level architecture singleton should not have danglin input slots. Use Sinks and Sources for external communication.")
    
    def register_output_slot(self, output_slot_id):
        raise CircuitError("The top-level architecture singleton should not have danglin output slots. Use Sinks and Sources for external communication.")
    