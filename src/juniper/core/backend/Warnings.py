

import logging

logger = logging.getLogger(__name__)
class JuniperWarning(Warning):
    pass


class CompilerWarning(JuniperWarning):
    pass

class ShapeInferenceWarning(CompilerWarning):
    pass

class TypeInferenceWarning(CompilerWarning):
    pass

class LoadBufferWarning(CompilerWarning):
    pass

class SaveBufferWarning(CompilerWarning):
    pass


class EngineWarning(JuniperWarning):
    pass

class NotCompiledWarning(EngineWarning):
    pass


class CircuitWarning(JuniperWarning):
    pass

class CircuitConnectionWarning(CircuitWarning):
    pass


class TCPWarning(JuniperWarning):
    pass

class RecordingWarning(JuniperWarning):
    pass


class JuniperConfigurationWarning(JuniperWarning):
    pass

class JuniperUserWarning(JuniperWarning):
    pass
