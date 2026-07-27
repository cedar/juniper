# Backend API

Most applications use the backend through `Architecture.compile()` and `Architecture.run_simulation()`. The backend is responsible for turning a graph into a JAX-executable simulation.

## Compiler

The public compiler function is available as `juniper.compile(circuit)`. It traverses the architecture, including nested circuits, and returns a `CompiledCircuit`. During compilation it validates connections, resolves element paths, infers shapes and dtypes, collects sources and sinks, identifies dynamic elements, and stores the execution order, compute kernels, initial runtime state, and current runtime state.

```python
compiled = jp.compile(arch)
```

`CompileInfo` contains:

| Field | Description |
|-------|-------------|
| `circuit` | Root circuit that was compiled. |
| `compiled_elements` | Mapping from path tuples to `ElementRef` objects. |
| `dynamic` / `static` | Elements with and without evolving runtime state. |
| `sources` / `sinks` | Runtime I/O endpoints. |
| `kernel_map` | Ordered path-to-kernel mapping used by the simulation runtime. |

## Compiled Circuit

`CompiledCircuit` is the explicit runtime object used by the simulation functions. It stores the compiled metadata, initial state, current state, PRNG tree, dynamic PRNG slots, and static PRNG key.

Most applications do not need to call `compile(...)` manually because `Architecture.compile()` does it and stores the result at `arch.runtime`. Use the function-based API when you want lower-level control:

```python
compiled = jp.compile(arch)

jp.load_buffers(compiled)
jp.open_connections(compiled)
jp.trace(compiled, warmup=1)
jp.reset_state(compiled)

recording, timing = jp.run_simulation(
    compiled,
    num_steps=100,
    steps_to_record=["field", "field.activation"],
)

jp.save_buffers(compiled)
jp.close_connections(compiled)
```

Available simulation functions:

| Function | Description |
|----------|-------------|
| `CompiledCircuit.from_circuit(circuit)` | Compile a circuit and return a compiled circuit. |
| `CompiledCircuit.from_compile_info(runtime_state, compile_info)` | Create a compiled circuit from existing compiler internals. |
| `trace(compiled, warmup=0)` | Trace the JAX tick and optionally run warmup ticks. |
| `init_prng(compiled)` | Initialize the PRNG tree and dynamic PRNG slots. |
| `refresh_prng(compiled)` | Refresh keys for dynamic elements before a tick. |
| `load_buffers(compiled)` | Load permanent buffers into runtime state. |
| `save_buffers(compiled)` | Save permanent buffers from runtime state. |
| `reset_state(compiled)` | Restore the compiled circuit to its initial state. |
| `open_connections(compiled)` | Open runtime I/O endpoints. |
| `close_connections(compiled)` | Close runtime I/O endpoints. |
| `run_simulation(compiled, num_steps, ...)` | Run fixed-step simulation and return `(Recording, TimingInfo)`. |

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
