# Frontend API

The frontend is the graph-building layer. Most user code interacts with the classes re-exported from `juniper`.

## Architecture

`get_arch(name=None)` returns the singleton top-level `Architecture`. `delete_arch()` clears the singleton and is useful in notebooks and tests.

`Architecture` adds runtime methods to `Circuit`:

| Method | Description |
|--------|-------------|
| `compile(warmup=0, print_compile_info=False, load_buffer=False)` | Compile the graph, allocate runtime state, optionally load permanent buffers, and trace the JAX tick. |
| `run_simulation(num_steps, steps_to_record=[], print_timing=True, save_buffer=False)` | Run fixed-step simulation and return `(Recording, TimingInfo)`. |
| `reset_state()` | Restore runtime state to the post-compilation initial state. |
| `close_connections()` | Close runtime I/O endpoints such as TCP workers. |
| `set_arch_name(name)` | Change the architecture name used in paths and buffer files. |

## Elements And Steps

`Element` is the base class for graph nodes. It owns input and output slots, stores parameters, and carries compiler metadata.

`Step` is the base class for computation nodes. A regular step registers default `in0` and `out0` slots. Steps can register additional slots and buffers.

`Source` is a step without input slots. It provides data to the runtime before each tick. `Sink` receives data after each tick.

## Slots And Connections

Slots are named endpoints on elements. Default slots are `in0` and `out0`.

```python
source >> step
source.out0 >> step.in0
step_b << step_a
source >> "field.in0"
```

The connection operators accept elements, slot objects, or string paths in the current circuit. Input aggregation is sum by default. A step may define a different aggregation rule, such as product aggregation in `ComponentMultiply`.

## Circuits

`Circuit` is an element that contains other elements. Use it as a context manager to build nested graphs:

```python
with jp.Circuit("preprocess") as preprocess:
    preprocess.register_input_slot("in0")
    preprocess.register_output_slot("out0")
    gain = jp.StaticGain("gain", 0.5)
    preprocess.in0 >> gain >> preprocess.out0
```

Circuit paths are represented with dots, for example `preprocess.gain.out0`.

## Configurable

`Configurable` stores validated parameter dictionaries for helper objects and elements. Public configurable helpers include `Gaussian`, `LateralKernel`, `Sigmoid`, `FrameGraph`, and `Transform`.

## Logging

JUNIPER uses Python logging. Optional setup helpers are available:

```python
import logging
import juniper as jp

jp.init_logging(level=logging.INFO)
jp.init_logging_to_file("juniper.log", level=logging.DEBUG)
```
