# Building Architectures

A JUNIPER architecture is a top-level `Architecture`, which is also a `Circuit`: a directed graph of elements connected through named slots. Steps perform computation, sources push external data into the runtime state, sinks pull data out, and nested circuits package reusable subgraphs.

## Top-Level Architecture

Use the singleton helpers from `juniper` for the usual workflow:

```python
from juniper import get_arch, delete_arch

delete_arch()          # useful in tests or notebooks
arch = get_arch("demo")
```

When an `Architecture` exists, newly created steps are registered in the current circuit automatically. The top-level architecture cannot have dangling input or output slots; use sources and sinks for external communication.

## Creating Steps

Most steps now take explicit constructor parameters:

```python
from juniper import CustomInput, StaticGain, Sum

signal = CustomInput("signal", shape=(32,))
gain = StaticGain("gain", factor=2.0)
merge = Sum("merge")
```

Every element name must be unique in its parent circuit and must not contain dots. Configurable helper objects such as `Gaussian` and `LateralKernel` still take dictionaries because they are not graph elements.

## Connecting Elements

Every regular step has default slots `in0` and `out0`. Additional slots are exposed as attributes with their slot names. The `>>` operator connects output to input and returns the right-hand object, enabling chains.

```python
signal >> gain >> merge

# Equivalent reverse connection.
gain << signal

# Explicit slots.
step_a.out0 >> step_b.in1

# String paths in the current circuit.
step_a >> "step_b.in0"
```

Multiple incoming connections are summed by default. `Sum` allows unlimited inputs on `in0`; `ComponentMultiply` also allows unlimited inputs but aggregates by product. Slots enforce their maximum incoming connection count and duplicate connections raise `CircuitConnectionError`.

## Nested Circuits

Use `Circuit` as a context manager to define reusable subgraphs. Register circuit input/output slots, connect internals, and then connect the circuit like any other element.

```python
from juniper import Circuit, StaticGain, get_arch

arch = get_arch()
with Circuit("double") as double:
    double.register_input_slot("in0")
    double.register_output_slot("out0")
    gain = StaticGain("gain", factor=2.0)
    double.in0 >> gain >> double.out0

# double can now be connected in the parent circuit.
```

Nested element paths use dots, for example `double.gain`. Recordings and compile errors use those paths.

## Circuit Subclasses

Reusable circuits can also be packaged as subclasses and imported like normal step classes. See [Circuit Subclasses](circuit-subclasses.md) for the full pattern.

## Compile And Run

Compile after the graph is complete. Compilation gathers graph metadata, infers shapes and dtypes, builds runtime state, opens source/sink connections, traces the JAX tick, and optionally performs warmup steps.

```python
arch.compile(warmup=1, print_compile_info=True, load_buffer=False)
recording, timing = arch.run_simulation(
    num_steps=100,
    steps_to_record=["field", "field.out0"],
    print_timing=True,
    save_buffer=False,
)
```

`run_simulation` returns `(Recording, TimingInfo)`. `TimingInfo` contains total time plus per-tick `prng`, `gpu_push`, `tick`, `gpu_pull`, and optional `buffer` timings.

There is no public `arch.tick()` method in the current API; run fixed numbers of ticks through `run_simulation`.

## Recording Data

Record targets may be strings, elements, slots, or buffers. The returned `Recording` object supports filtering and persistence:

```python
rec, timing = arch.run_simulation(50, steps_to_record=["field.out0"])
field_only = rec.get_at_element("field.out0")
first_ten = rec.get_in_step_interval((0, 10))
run_dir = rec.save_to_file("recordings")
loaded = type(rec).load_from_file(run_dir)
rec.plot(keys=["field.out0"], idx_interval=(0, 50))
```

Saved recordings use one pickle file per time step plus a `manifest.json`, so additional batches can append to the same run directory when the recorded keys match.

## Sources And Sinks

Sources write CPU-side or external data into the runtime state before each tick. `CustomInput` can be updated with `set_data`; `TCPReader` uses a shared-memory worker process. Sinks receive data after each tick. `TCPWriter` sends data through the TCP worker and `StaticDebug` is useful to force static branches to be computed.

Call `arch.close_connections()` if a long-running process needs explicit cleanup after TCP runs.

## Buffers

Steps can register internal buffers with `register_buffer(buf_id, shape, permanent=False)`. Dynamic steps use buffers for evolving state such as activation, weights, local timers, or fixation state. Permanent buffers are loaded during `compile(load_buffer=True)` and saved after `run_simulation(save_buffer=True)`.

## Extending JUNIPER

Subclass `Step`, `Source`, or `Sink`, define a JAX-compatible `compute_kernel`, and override `infer_output_shapes` or `infer_output_dtypes` whenever the default “mirror input slot shape and default dtype” behavior is not enough. Kernel outputs must return exactly the compiled state keys with stable shapes; mismatches raise `EngineError`.
