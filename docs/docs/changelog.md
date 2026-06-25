# Changelog

This page summarizes the major changes and additions introduced by the recent JUNIPER refactor. 

## Major Refactor

### Public API

- Step constructors now use explicit parameters instead of a single parameter dictionary.
- Most commonly used classes and error types are re-exported from `juniper`.
- Configurable helper objects such as `Gaussian`, `LateralKernel`, `FrameGraph`, and `Transform` still use parameter dictionaries.
- Top-level architecture access is centered on `get_arch()` and `delete_arch()`.
- Simulation is run through `Architecture.run_simulation(...)`; there is no public `arch.tick()` method in the current API.
- Logging setup helpers are now exposed as `init_logging(...)` for console output and `init_logging_to_file(...)` for file output.

### Frontend And Architecture Building

- Added reusable nested `Circuit` support with context-manager construction.
- Added named slot connection support through slot attributes and string paths.
- Improved connection validation for duplicate connections, invalid endpoints, and max fan-in limits.
- Element paths now carry nested circuit structure and are used by recordings, compile info, and errors.

### Backend And Runtime

- Added a clearer backend split around `Compiler`, `Engine`, `RuntimeState`, `CompileInfo`, and `Recording`.
- Compilation now gathers static, dynamic, source, sink, kernel, shape, and dtype metadata before runtime execution.
- Runtime state is stored as a flat path-keyed tree with helpers for reading slots, writing sources, and recording targets.
- The engine manages source pushes, PRNG key updates, JAX-jitted ticks, sink pulls, recordings, timing, and permanent buffer save/load.
- Kernel state returns are normalized against the compiled state contract and raise `EngineError` on missing keys or shape mismatches.

### Diagnostics

- Added domain-specific exception and warning hierarchies.
- Compile failures now include traced upstream causes where possible, including unresolved inputs, unresolved buffers, cycles, shape inference failures, and dtype inference failures.
- Added dedicated recording load/save errors and TCP errors.
- Added library-wide Python logging hooks for compiler, engine, frontend, TCP, and step diagnostics.

### Recording

- `run_simulation` now returns a `Recording` object and timing dictionary.
- Record targets can be strings, elements, slots, or buffers.
- `Recording` supports filtering by element, filtering by time interval, slicing, appending, plotting, saving, and loading.
- Saved recordings use a recording manifest plus one pickle file per time step, allowing compatible batches to append to an existing run directory.
