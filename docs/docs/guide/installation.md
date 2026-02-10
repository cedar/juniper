# Installation

## Requirements

- Python >= 3.9
- A CUDA-capable GPU is recommended but not required (CPU fallback is available)

## Install from Source

Clone the repository and install in editable mode:

```bash
pip install -e /path/to/juniper
```

### GPU Support (CUDA)

For GPU acceleration, install with the CUDA extra. Make sure your system has a compatible NVIDIA driver.

```bash
# CUDA 12
pip install -e /path/to/juniper[cuda12]
```

Refer to the [JAX installation guide](https://github.com/google/jax#installation) to verify GPU support for your platform.

## Dependencies

JUNIPER depends on:

- **jax** / **jaxlib** -- GPU computation and JIT compilation
- **matplotlib** -- plotting and visualization
- **flax** / **flaxmodels** -- neural network support (used by the DNN step)

All dependencies are installed automatically via `pip`.

## Verify Installation

```bash
python -c "import juniper; print('JUNIPER installed successfully')"
```

To confirm GPU availability:

```python
import jax
print(jax.devices())  # Should list GPU device(s)
```
