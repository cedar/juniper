

class JuniperError(Exception):
    pass


class CompilerError(JuniperError):
    pass

class ShapeInferenceError(CompilerError):
    pass

class TypeInferenceError(CompilerError):
    pass

class LoadBufferError(CompilerError):
    pass

class SaveBufferError(CompilerError):
    pass


class EngineError(JuniperError):
    pass

class NotCompiledError(EngineError):
    pass


class CircuitError(JuniperError):
    pass

class CircuitConnectionError(CircuitError):
    pass


class TCPError(JuniperError):
    pass

class RecordingError(JuniperError):
    pass


class JuniperConfigurationError(JuniperError):
    pass

class JuniperUserError(JuniperError):
    pass
