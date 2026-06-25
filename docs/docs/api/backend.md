# Backend API

The backend compiles frontend circuits into a runtime representation and executes them through JAX. Most applications use it indirectly through `Architecture.compile()` and `Architecture.run_simulation()`.

## Compiler

`Compiler` traverses circuits, resolves connectivity, infers output shapes and dtypes, and builds a `CompileInfo` object which holds meta data information regardings the steps dynamic/static/sink/source properties and its kernels. Compile failures include traced upstream causes when possible, including unresolved inputs, unresolved buffers, cycles, shape inference failures, and dtype inference failures.

`CompileInfo` contains:

| Field | Purpose |
|-------|---------|
| `circuit` | Compiled root circuit. |
| `compiled_elements` | Mapping from element path tuple to `ElementRef`. |
| `dynamic` / `static` | Ordered refs for dynamic and static computation. |
| `sources` / `sinks` | Runtime I/O endpoints. |
| `kernel_map` | Compiled path-to-kernel execution order. |

## Engine

`Engine` owns runtime execution. It allocates `RuntimeState`, manages PRNG keys, opens and closes source/sink connections, jits `_tick`, pushes source data before each tick, pulls sink and recording data after each tick, and saves/loads permanent buffers.

`run_simulation` returns a `Recording` and a timing dictionary with `total`, `prng`, `gpu_push`, `gpu_pull`, `tick`, `buffer`, and `num_steps`.

## RuntimeState

`RuntimeState` wraps the flat state tree keyed by element path. It can read slots, write source outputs, trace nested state, record targets, and copy the initialized state. Runtime kernels must preserve compiled keys and shapes; returned arrays are converted back to the compiled dtype.

## Recording

`Recording(recording, keys)` stores a list of time-step rows. Useful methods include `get_at_element`, `get_at_elements`, `get_at_step`, `get_in_step_interval`, `slice`, `append`, `save_to_file`, `load_from_file`, and `plot`.

## TCP Backend

`TCPReader` and `TCPWriter` launch a `TCPWorker` process and communicate through shared memory. Data is serialized in an OpenCV-compatible matrix format with CRC checks, heartbeat handling, retry delays, and shape/dtype validation.
