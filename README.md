# JUNIPER

Welcome to JUNIPER, the GPU Accelerated Python Implementation of CEDAR

## Requirements

- Python >= 3.9
- JAX and jaxlib
- A CUDA-capable GPU is recommended for accelerated runs, but CPU execution is supported by JAX.

Check available JAX devices with:

```python
import jax
print(jax.devices())
```

## Installation

Once JUNIPER is published to PyPI, install it directly:

```bash
pip install juniper
```

For CUDA-enabled JAX builds, install the matching extra:

```bash
pip install "juniper[cuda12]"
pip install "juniper[cuda13]"
```

For development from a local checkout:

```bash
git clone https://github.com/cedar/juniper.git
cd juniper
pip install -e .
```

Development and documentation extras are available:

```bash
pip install -e ".[dev]"
pip install -e ".[docs]"
```

The documentation uses Material for MkDocs:

```bash
mkdocs build --config-file docs/mkdocs.yml
```

## Quick Start

```python
import numpy as np
from juniper import CustomInput, Gaussian, NeuralField, StaticGain, get_arch

arch = get_arch("demo")

source = CustomInput("source", shape=(50,))
source.set_data(np.ones((50,), dtype=np.float32))

gain = StaticGain("gain", factor=0.8)
field = NeuralField(
    "field",
    shape=(50,),
    resting_level=-5.0,
    global_inhibition=-0.01,
    tau=0.1,
    input_noise_gain=0.1,
    lateral_kernel=Gaussian({
        "shape": (50,),
        "sigma": (3,),
        "amplitude": 5.0,
        "normalized": True,
        "factorized": False,
    }),
)

source >> gain >> field

arch.compile(warmup=1)
recording, timing = arch.run_simulation(
    num_steps=100,
    steps_to_record=[field, "field.activation"],
)
```

## Core Concepts

- **Architecture**: the top-level circuit returned by `get_arch()`.
- **Steps**: computation nodes with named input/output slots, such as `StaticGain`, `NeuralField`, `Sum`, `ColorConversion`, and robotics conversion steps.
- **Sources and sinks**: runtime I/O endpoints such as `CustomInput`, `TCPReader`, `TCPWriter`, and `StaticDebug`.
- **Circuits**: reusable nested graphs. You can create them inline or subclass `Circuit` and instantiate them like normal steps.
- **Configurables**: helper objects such as `Gaussian`, `LateralKernel`, `FrameGraph`, and `Transform`.
- **Recording**: `run_simulation` returns a `Recording` object with slicing, plotting, save, and load helpers.

## Command-Line Usage

JUNIPER also includes `run.py` for running architecture files from the command line:

```bash
python run.py path/to/architecture.py --num_ticks 500 --recording field
```
