from .configurables.Circuit import Circuit


class Architecture(Circuit):
    def __init__(self, name : str = "architecture"):
        if Circuit._current is not None:
            raise Exception("Parent circuit already exists. Use this class only to initialize the top-level architecture.")
        else:
            Circuit._current = self
        super().__init__(name = name)
