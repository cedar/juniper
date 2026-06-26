# Building Architectures

A JUNIPER architecture is a directed graph of connected elements. Sources provide data, steps compute outputs, sinks consume outputs, and circuits group elements into reusable subgraphs.

The usual workflow is:

1. Create or retrieve the top-level architecture.
2. Instantiate sources, steps, sinks, and optional nested circuits.
3. Connect elements through slots.
4. Compile the architecture.
5. Run a fixed number of simulation ticks.
6. Inspect, plot, or save recordings.

## Top-Level Architecture

```python
import juniper as jp

arch = jp.get_arch("demo")
```

`get_arch()` returns a singleton top-level `Architecture`. New elements are registered in the currently active circuit. In scripts and notebooks, use `delete_arch()` before building a separate architecture in the same Python process.

```python
jp.delete_arch()
arch = jp.get_arch("second_demo")
```

The top-level architecture cannot have public input or output slots. Use sources and sinks for communication with Python or external processes.

## Steps And Sources

```python
import numpy as np
import juniper as jp

source = jp.CustomInput("source", shape=(32,))
source.set_data(np.ones((32,), dtype=np.float32))

gain = jp.StaticGain("gain", factor=0.5)
field = jp.NeuralField("field", shape=(32,), resting_level=-5.0)
```

Element names must be unique within their parent circuit and must not contain dots. Dots are reserved for nested paths such as `vision.field.activation`.

Most graph elements have default slots named `in0` and `out0`. Additional slots are exposed as attributes, for example `step.in1`, `step.out1`, or `camera.viewport_center`.

## Connections

Use `>>` to connect an output to an input. The operator returns the right-hand object, so chains are supported.

```python
source >> gain >> field
```

Equivalent forms are available when you need explicit slots or string references:

```python
gain << source
source.out0 >> field.in0
source >> "field.in0"
```

Incoming values are summed by default. Steps can define another aggregation rule; `ComponentMultiply` multiplies all incoming values on `in0`. Slots also enforce their maximum number of incoming connections.

## Nested Circuits

A `Circuit` is both a graph element and a container. Use it when several steps should behave as one reusable component.

```python
with jp.Circuit("double") as double:
    double.register_input_slot("in0")
    double.register_output_slot("out0")

    gain = jp.StaticGain("gain", factor=2.0)
    double.in0 >> gain >> double.out0

source >> double
```

Nested paths use dot notation. The internal gain above can be recorded as `double.gain` or `double.gain.out0`.

Reusable components can also be written as subclasses. See [Circuit Subclasses](circuit-subclasses.md).

## Compilation

Compile after the graph is complete:

```python
arch.compile(warmup=1, print_compile_info=True, load_buffer=False)
```

Compilation performs the background work needed for fast simulation:

- traverses the graph and nested circuits,
- validates connections and input limits,
- infers output shapes and dtypes,
- creates the runtime state tree,
- registers static steps, dynamic steps, sources, sinks, and kernels,
- loads permanent buffers when requested,
- opens runtime I/O endpoints,
- traces the JAX tick function,
- optionally runs warmup ticks and resets the state afterward.

Keep shapes and dtypes stable after compilation. Changing the shape or dtype of a source can force JAX retracing or raise a runtime error.

## Simulation

Run simulations through `run_simulation`:

```python
recording, timing = arch.run_simulation(
    num_steps=100,
    steps_to_record=["field", "field.activation"],
    print_timing=True,
    save_buffer=False,
)
```

Each tick performs the following steps:

1. Push source data into runtime state.
2. Update PRNG keys for dynamic elements.
3. Execute the JAX-jitted tick kernel.
4. Pull sink data and requested recordings back to Python.
5. Save permanent buffers at the end when `save_buffer=True`.

Use `arch.reset_state()` to return to the post-compilation initial state. Use `arch.close_connections()` to close TCP workers or other runtime connections.

## Recording, Plotting, And Saving

Recording targets can be strings, elements, slots, or buffers.

```python
rec, _ = arch.run_simulation(
    50,
    steps_to_record=["field", "field.out0", "field.activation"],
)

field_rec = rec.get_at_element("field")
first_ten = rec.get_in_step_interval((0, 10))
subset = rec.slice(["field.activation"], (10, 30))
fig = rec.plot(keys=["field.activation"], snapshot_indices=[0, 49])
run_dir = rec.save_to_file("recordings")
loaded = type(rec).load_from_file(run_dir)
```

`Recording.plot` plots scalar recordings as time courses and non-scalar recordings as snapshots. For scalar groups, pass `group_keys=[["loss", "reward"]]`.

Saved recordings use a `manifest.json` file plus one pickle file per time step. Additional batches can append to an existing run directory when the recorded keys match.

## Persistent Buffers

Some dynamic steps store internal buffers, such as field activations or learned weights. Buffers marked permanent are saved with `run_simulation(save_buffer=True)` and loaded with `compile(load_buffer=True)`.

This is useful for learned connection weights in steps such as `HebbianConnection` and `BCMConnection`.

## Extending JUNIPER

Custom elements usually subclass `Step`, `Source`, or `Sink`.

A custom step should:

- register any additional slots or buffers in `__init__`,
- provide a JAX-compatible `compute_kernel(input, state, **kwargs)`,
- return a dictionary containing all output slots and updated buffers,
- override `infer_output_shapes` or `infer_output_dtypes` when the default behavior is not sufficient.

Kernel outputs must keep the compiled state keys, shapes, and dtypes stable.
