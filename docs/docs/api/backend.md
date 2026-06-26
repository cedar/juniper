# Backend API

Most applications use the backend through `Architecture.compile()` and `Architecture.run_simulation()`. The backend is responsible for turning a graph into a JAX-executable simulation.

## Compiler

The compiler traverses the architecture, including nested circuits, and builds a `CompileInfo` object. During compilation it validates connections, resolves element paths, infers shapes and dtypes, collects sources and sinks, identifies dynamic elements, and stores the execution order and compute kernels.

`CompileInfo` contains:

| Field | Description |
|-------|-------------|
| `circuit` | Root circuit that was compiled. |
| `compiled_elements` | Mapping from path tuples to `ElementRef` objects. |
| `dynamic` / `static` | Elements with and without evolving runtime state. |
| `sources` / `sinks` | Runtime I/O endpoints. |
| `kernel_map` | Ordered path-to-kernel mapping used by the engine. |

## Engine

`Engine` owns the simulation loop. It allocates runtime state, manages PRNG keys, opens and closes runtime I/O, executes the JAX-jitted tick, records requested values, and saves or loads permanent buffers.

A simulation tick does this work:

1. Copy source data into runtime state.
2. Generate PRNG keys for dynamic elements.
3. Execute the compiled JAX tick.
4. Copy sink outputs and recordings back to Python.

`run_simulation` returns a `Recording` and a timing dictionary with `total`, `prng`, `gpu_push`, `gpu_pull`, `tick`, `buffer`, and `num_steps`.

## Runtime State

Runtime state is a flat tree keyed by element path. Each entry stores output slots and buffers for one compiled element. Kernels must return exactly the state entries they own, with stable shapes and compatible dtypes.

## Recording

`Recording(recording, keys)` stores a time-major list of recorded arrays. Useful methods are:

| Method | Description |
|--------|-------------|
| `get_at_element(key)` | Keep one recording target. |
| `get_at_elements(keys)` | Keep several targets. |
| `get_at_step(step_idx)` | Keep one simulation time step. |
| `get_in_step_interval((start, stop))` | Keep a time interval. |
| `slice(keys, interval)` | Filter targets and time interval together. |
| `append(recording)` | Append compatible recordings. |
| `save_to_file(path, run_dir=None)` | Save as manifest plus per-step pickle files. |
| `load_from_file(run_dir)` | Load a saved recording. |
| `plot(...)` | Plot scalar tikme courses and array snapshots. |

## TCP Runtime

`TCPReader` and `TCPWriter` use a worker process and shared memory. The worker handles socket setup, retry delays, heartbeat state, data serialization, CRC checks, and shape/dtype validation. Call `arch.close_connections()` after TCP simulations when your process continues running.
