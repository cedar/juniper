# Errors And Warnings

JUNIPER defines domain-specific exceptions and warnings; the common public types are re-exported from `juniper`, and the full hierarchy is available under `juniper.core.backend.Exceptions` and `juniper.core.backend.Warnings`. Catch the narrowest type that matches the layer you are handling.

## Exceptions

| Type | Raised For |
|------|------------|
| `JuniperError` | Base class for JUNIPER errors. |
| `CompilerError` | General compilation failures. |
| `ShapeInferenceError` | A step cannot infer an output shape. |
| `TypeInferenceError` | A step cannot infer an output dtype. |
| `LoadBufferError` | Persistent buffer load failure. |
| `SaveBufferError` | Persistent buffer save failure. |
| `EngineError` | Runtime, tracing, or kernel state-contract failure. |
| `NotCompiledError` | Simulation requested before compilation. |
| `CircuitError` | Invalid circuit/element/slot structure. |
| `CircuitConnectionError` | Invalid, duplicate, or over-capacity connection. |
| `TCPError` | TCP worker/shape/protocol failure. |
| `RecordingError` | Recording access or append mismatch. |
| `LoadRecordingError` | Saved recording is missing or inconsistent. |
| `SaveRecordingError` | Recording cannot be written safely. |
| `JuniperConfigurationError` | Invalid constructor/configurable parameters. |
| `JuniperUserError` | User-facing API misuse, such as invalid element names. |

## Warnings

Warning classes mirror the exception hierarchy where useful: `JuniperWarning`, `CompilerWarning`, `ShapeInferenceWarning`, `TypeInferenceWarning`, `LoadBufferWarning`, `SaveBufferWarning`, `EngineWarning`, `NotCompiledWarning`, `CircuitWarning`, `CircuitConnectionWarning`, `TCPWarning`, `RecordingWarning`, `JuniperConfigurationWarning`, and `JuniperUserWarning`.

Warnings are used for recoverable situations such as defaulting missing centers in Gaussian inputs or non-fatal compiler/runtime diagnostics.
