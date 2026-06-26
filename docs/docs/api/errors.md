# Errors And Warnings

JUNIPER exposes domain-specific exception and warning classes. The common classes are re-exported from `juniper`.

## Exceptions

| Type | Typical Cause |
|------|---------------|
| `JuniperError` | Base class for JUNIPER exceptions. |
| `JuniperUserError` | User-facing API misuse, such as invalid element names. |
| `JuniperConfigurationError` | Invalid constructor or configurable parameters. |
| `CircuitError` | Invalid element, slot, or circuit structure. |
| `CircuitConnectionError` | Invalid connection, duplicate connection, or exceeded connection limits. |
| `CompilerError` | General compilation failure. |
| `ShapeInferenceError` | A step cannot infer an output shape. |
| `TypeInferenceError` | A step cannot infer an output dtype. |
| `EngineError` | Runtime, tracing, or kernel state-contract failure. |
| `NotCompiledError` | Simulation requested before compilation. |
| `RecordingError` | Invalid recording access or incompatible append. |
| `LoadRecordingError` | Saved recording is missing or inconsistent. |
| `SaveRecordingError` | Recording cannot be written safely. |
| `LoadBufferError` | Persistent buffer load failure. |
| `SaveBufferError` | Persistent buffer save failure. |
| `TCPError` | TCP worker, protocol, shape, or dtype failure. |

## Warnings

Warning classes mirror the main exception categories: `JuniperWarning`, `JuniperUserWarning`, `JuniperConfigurationWarning`, `CircuitWarning`, `CircuitConnectionWarning`, `CompilerWarning`, `ShapeInferenceWarning`, `TypeInferenceWarning`, `EngineWarning`, `NotCompiledWarning`, `RecordingWarning`, `LoadBufferWarning`, `SaveBufferWarning`, and `TCPWarning`.

Configure Python logging or warnings in your application when you want to capture diagnostics during compilation or simulation.
