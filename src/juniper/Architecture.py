from .configurables.Circuit import Circuit
from .configurables import CircuitContext

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
        if Circuit._current is not None:
            raise Exception("Parent circuit already exists. Use this class only to initialize the top-level architecture.")
        else:
            Circuit._current = self
            CircuitContext.set_current(self)
        super().__init__(name = name)

    def set_arch_name(self, name : str):
        self._name = name
