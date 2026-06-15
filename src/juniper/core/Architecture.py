from .Circuit import Circuit
from . import CircuitContext
from .Engine import Engine
from .Engine import Recording
from .Engine import TimingInfo

_architecture_singleton = None

def get_arch(name=None):
    global _architecture_singleton
    if _architecture_singleton is None:
        _architecture_singleton = Architecture() if name is None else Architecture(name=name)
    return _architecture_singleton

def delete_arch():
    global _architecture_singleton
    _architecture_singleton = None
    Circuit._current = None
    CircuitContext.set_current(None)

class Architecture(Circuit):
    def __init__(self, name : str = "architecture"):
        """A singleton instance for the top-level circuit. Cannot have input or output slots.\n
        The architecture class also includes useful functions for comilation and simulation without manually having to call engine and compiler."""
        if Circuit._current is not None:
            raise Exception("Parent circuit already exists. Use this class only to initialize the top-level architecture.")
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

    def set_input(self, input_slot_id, dest_slot, max_incoming_connections = 1):
        raise Exception("The top-level architecture singleton should not have danglin input slots. Use Sinks and Sources for external communication.")
    
    def set_output(self, output_slot_id, source_slot):
        raise Exception("The top-level architecture singleton should not have danglin output slots. Use Sinks and Sources for external communication.")
    