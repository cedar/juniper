# Frontend API

The frontend is the user-facing graph-building layer. It lives under `juniper.core.frontend` and is re-exported through `juniper` where appropriate.

## Architecture

`get_arch(name: str | None = None)` returns the singleton top-level `Architecture`. `delete_arch()` clears it, which is useful in tests and notebooks.

`Architecture` inherits from `Circuit` and adds:

| Method | Purpose |
|--------|---------|
| `compile(warmup=0, print_compile_info=False, load_buffer=False)` | Compile the graph and initialize runtime state. |
| `run_simulation(num_steps, steps_to_record=[], print_timing=True, save_buffer=False)` | Run fixed-step simulation and return `(Recording, TimingInfo)`. |
| `reset_state()` | Restore runtime state to post-compilation initial state. |
| `close_connections()` | Close source/sink connections and worker processes. |

## Logging

JUNIPER uses Python's standard `logging` module throughout the frontend, backend, TCP workers, and step library. This version exposes small setup helpers from `juniper`:

```python
import logging
from juniper import init_logging, init_logging_to_file

init_logging(level=logging.INFO)
init_logging_to_file("juniper.log", level=logging.DEBUG)
```

`init_logging(...)` adds a console handler. `init_logging_to_file(...)` appends logs to a file. These helpers are optional; applications can also configure Python logging directly.

## Element Model

`Element` is the base for graph nodes. It validates names, stores parameters, owns input/output slot maps, and carries compiler metadata: `is_dynamic`, `is_source`, `is_sink`, `needs_input_connections`, `input_aggregation`, and `is_compiled`.

`Step` registers default `in0` and `out0` slots and can register internal buffers. `Source` and `Sink` specialize step behavior for external data exchange.

## Slots And Connections

`Slot` objects belong to elements and are named with strings such as `in0`, `out0`, `learn_node`, or `in1`. Connections are created through `>>`, `<<`, or `Circuit.connect_to(source, dest)`. Connectables can be elements, slots, or strings like `"field.in0"`.

Input aggregation defaults to sum. Steps can change this; `ComponentMultiply` sets aggregation to product. `max_incoming_connections` prevents invalid fan-in unless a step explicitly raises the limit.

## Circuits

`Circuit` is both an element and a container. Use it as a context manager to set the current parent circuit while declaring internal elements. On exit, `define_circuit_structure()` generates the pass-through kernel and registers the circuit in its parent. Circuit slots bridge parent connections into and out of the nested graph.

## Configurable

`Configurable` stores `_params`, adds `name`, and validates mandatory parameters. It is used by elements and by helper objects such as `Gaussian`, `LateralKernel`, `Sigmoid`, `FrameGraph`, and `Transform`.
