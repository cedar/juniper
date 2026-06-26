# JUNIPER

Welcome to JUNIPER, the GPU Accelerated Python Implementation of CEDAR

## Requirements

- Python 3.9 or newer
- JAX and jaxlib
- matplotlib for plotting
- flax and flaxmodels for the `DNN` step

JAX can run on CPU. A CUDA-capable GPU is recommended for larger simulations.

Check the devices visible to JAX with:

```python
import jax
print(jax.devices())
```

## Installation From Source

From a local checkout:

```bash
pip install -e /path/to/juniper
```

For CUDA-enabled JAX builds, install the matching optional dependency:

```bash
pip install -e "/path/to/juniper[cuda12]"
pip install -e "/path/to/juniper[cuda13]"
```

## Quick Start

```python
import numpy as np
import juniper as jp

arch = jp.get_arch("demo")

source = jp.CustomInput("source", shape=(50,))
source.set_data(np.ones((50,), dtype=np.float32))

field = jp.NeuralField(
    "field",
    shape=(50,),
    resting_level=-5.0,
    global_inhibition=-0.01,
    tau=0.1,
    input_noise_gain=0.0,
)

source >> field

arch.compile(warmup=1)
recording, timing = arch.run_simulation(
    num_steps=100,
    steps_to_record=["field", "field.activation"],
)

fig = recording.plot(keys=["field.activation"])
```

## Core Concepts

- **Architecture**: the top-level circuit returned by `get_arch()`.
- **Steps**: computation nodes with named input and output slots.
- **Sources**: nodes that provide external data to the simulation, such as `CustomInput`, `ImageLoader`, and `TCPReader`.
- **Sinks**: nodes that receive data from the simulation, such as `TCPWriter`.
- **Circuits**: reusable nested graphs built from steps and slots.
- **Configurables**: helper objects such as `Gaussian`, `LateralKernel`, `FrameGraph`, and `Transform`.
- **Recording**: the result object returned by `run_simulation`, with access, plotting, save, and load helpers.

## CLI

The command-line tool can execute architecture files that define `get_architecture()` or `get_architecture(args)`:

```bash
python run.py path/to/architecture.py --num_ticks 500 --recording field field.activation
```
