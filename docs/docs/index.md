# JUNIPER

**GPU-accelerated neural dynamics and signal-processing architectures in Python.**

JUNIPER builds directed computation graphs from small processing elements, compiles the graph into JAX kernels, and runs fixed-step simulations on CPU/GPU/TPU backends supported by JAX. The library is centered on Dynamic Field Theory, but the step library also covers array operations, image processing, robotics transformations, TCP I/O, recording, and reusable nested circuits.

## Current API Shape

Most step constructors now use explicit parameters instead of a single parameter dictionary:

```python
from juniper import GaussInput, NeuralField, StaticGain, Gaussian, get_arch

arch = get_arch()
source = GaussInput("input", shape=(50,), sigma=(3,), amplitude=5.0)
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
recording, timing = arch.run_simulation(100, steps_to_record=[field])
```

Configurables such as `Gaussian`, `LateralKernel`, `Sigmoid`, `FrameGraph`, and `Transform` still take parameter dictionaries.

## Highlights

- Explicit constructor API for steps and sources.
- `Circuit` context managers for reusable nested circuits.
- Frontend connection syntax with `>>`, `<<`, slot objects, and string paths.
- Backend compiler with shape/dtype inference and traceable compile-failure reports.
- JAX-jitted engine with runtime state, PRNG tree management, source/sink I/O, and persistent buffers.
- `Recording` utilities for slicing, plotting, saving, and loading simulation output.
- Expanded DFT, image-processing, robotics, TCP, and error/warning APIs.

## Documentation Overview

| Section | Description |
|---------|-------------|
| [Installation](guide/installation.md) | Installation and runtime dependencies. |
| [Building Architectures](guide/architecture.md) | Creating steps, circuits, connections, compilation, recording, and simulation. |
| [Command-Line Reference](guide/cli.md) | `run.py` command-line usage. |
| [Steps Reference](steps/index.md) | Built-in source, sink, DFT, array, image, robotics, and algebra steps. |
| [Configurables](configurables/index.md) | Kernels, sigmoid functions, and robotics frame helpers. |
| [API Reference](api/frontend.md) | Frontend, backend, recording, errors, and warnings. |
