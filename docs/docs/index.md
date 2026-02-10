# JUNIPER

**GPU Accelerated Python Implementation of CEDAR**

JUNIPER is a high-performance framework for simulating neural dynamics architectures on the GPU. Built on top of [JAX](https://github.com/google/jax), it provides a flexible Python API for constructing computational graphs of processing steps, compiling them, and running simulations with automatic GPU acceleration.

## What is JUNIPER?

JUNIPER implements the concepts of the CEDAR framework in Python, leveraging JAX for just-in-time (JIT) compilation and GPU execution. It is designed for researchers and engineers working with **Dynamic Field Theory (DFT)** and neural dynamics, but its modular step-based architecture makes it suitable for any signal-processing pipeline that benefits from GPU acceleration.

### Key Features

- **GPU acceleration** via JAX with automatic JIT compilation
- **Modular step-based architecture** -- compose complex processing pipelines from reusable building blocks
- **Pythonic connection syntax** using `>>` and `<<` operators to wire steps together
- **Rich library of built-in steps** covering algebra, array manipulation, neural fields, image processing, robotics, and I/O
- **Dynamic and static steps** -- efficient separation of time-varying (dynamic) and feed-forward (static) computation
- **Buffer save/restore** for persisting simulation state across runs

## Quick Start

```python
from juniper import GaussInput, NeuralField, Gaussian, StaticGain
from juniper.Architecture import get_arch

# Create steps
gi = GaussInput("input", {"shape": (50,), "sigma": (3,), "amplitude": 5})
gain = StaticGain("gain", {"factor": 0.8})
nf = NeuralField("field", {
    "shape": (50,), "resting_level": -5, "global_inhibition": -0.01,
    "tau": 0.1, "input_noise_gain": 0.1,
    "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0.5,
    "LateralKernel": Gaussian({"sigma": (3,), "amplitude": 5, "normalized": True, "max_shape": (50,)}),
})

# Connect steps using >> operator
gi >> gain >> nf

# Compile and run
arch = get_arch()
arch.compile()
for _ in range(100):
    arch.tick()
```

## Documentation Overview

| Section | Description |
|---------|-------------|
| [Installation](guide/installation.md) | How to install JUNIPER and its dependencies |
| [Building Architectures](guide/architecture.md) | How to create steps, connect them, and run simulations |
| [Command-Line Reference](guide/cli.md) | All `run.py` arguments |
| [Steps Reference](steps/index.md) | Complete reference for all built-in steps |
| [Configurables](configurables/index.md) | Configurable objects (Gaussian, LateralKernel, Sigmoid) |
