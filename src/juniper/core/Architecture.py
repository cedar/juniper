from __future__ import annotations
import logging
import sys
from .frontend.Circuit import Circuit
from .frontend import CircuitContext
from .backend.Compiler import compile as compile_circuit
from .backend.DataClasses import Recording
from .backend.DataClasses import RecKey
from .backend.DataClasses import TimingInfo
from .backend.Exceptions import CompilerError
from .backend.Exceptions import EngineError
from .backend.Exceptions import NotCompiledError
from .backend.Exceptions import CircuitError
from .backend.Simulation import SimulationRuntime
from .backend.Simulation import close_connections
from .backend.Simulation import load_buffers
from .backend.Simulation import open_connections
from .backend.Simulation import reset_state
from .backend.Simulation import run_simulation
from .backend.Simulation import trace
from ..util.util import timer


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
        self.runtime: SimulationRuntime | None = None

    def set_arch_name(self, name : str):
        self._name = name

    def compile(self, warmup : int = 0, print_compile_info : bool = False, load_buffer : bool = False) -> None:
        if self.is_compiled:
            raise CompilerError(f"The circuit {self.get_local_circuit_id()} is already compiled.")

        t_compile, (runtime_state, compile_info) = timer(compile_circuit)(self)
        self.runtime = SimulationRuntime.from_compiled_circuit(runtime_state, compile_info)

        if load_buffer:
            load_buffers(self.runtime)
            self.runtime.init_state = self.runtime.state.copy()
        open_connections(self.runtime)

        try:
            t_trace, _ = timer(trace)(self.runtime, warmup)
            reset_state(self.runtime)

            if print_compile_info:
                _print_compile_info(
                    self,
                    {
                        "t_compile": t_compile,
                        "t_trace": t_trace,
                        "N_static": len(compile_info.static),
                        "N_dynamic": len(compile_info.dynamic),
                        "N_total": len(compile_info.compiled_elements),
                        "N_warmup": warmup,
                    },
                )
        except Exception as e:
            logger.info((self.runtime.state.get_specs()))
            raise EngineError(
                "During Jax tracing and warmup an exception occured. "
                "The full state tree specs are written to logging.info:"
            ) from e

    def run_simulation(
            self,
            num_steps: int,
            steps_to_record: list[RecKey] = [],
            print_timing: bool = True,
            save_buffer: bool = False,
        )-> tuple[Recording, TimingInfo]:
        if self.runtime is None:
            raise NotCompiledError("Can't run simulation before compilation.")
        return run_simulation(runtime=self.runtime, num_steps=num_steps, steps_to_record=steps_to_record, print_timing=print_timing, save_buffer=save_buffer)

    def reset_state(self):
        if self.runtime is None:
            raise NotCompiledError("Can't reset state before compilation.")
        reset_state(self.runtime)

    def close_connections(self):
        if self.runtime is None:
            raise NotCompiledError("Can't close connections before compilation.")
        close_connections(self.runtime)

    def clean(self) -> None:
        super().clean()
        self.runtime = None

    def register_input_slot(self, input_slot_id, max_incoming_connections = 1):
        raise CircuitError("The top-level architecture singleton should not have danglin input slots. Use Sinks and Sources for external communication.")
    
    def register_output_slot(self, output_slot_id):
        raise CircuitError("The top-level architecture singleton should not have danglin output slots. Use Sinks and Sources for external communication.")


def _print_compile_info(circuit: Circuit, timing: TimingInfo) -> None:
    n_static = timing["N_static"]
    n_dynamic = timing["N_dynamic"]
    n_total = timing["N_total"]
    t_compile = timing["t_compile"]
    t_trace = timing["t_trace"]
    print(f"Compiled circuit '{circuit.get_local_circuit_id()}' with:")
    print(f"{n_static} static steps,")
    print(f"{n_dynamic} dynamic steps,")
    print(f"making a total of {n_total} steps.")
    print(f"{t_compile:6.2f} s for compilaton of initial state shapes and dtypes")
    print(f"{t_trace:6.2f} s for jax tracing of state and compute kernels")
    print("\n")
