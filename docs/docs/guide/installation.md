# Installation

## Requirements

JUNIPER requires Python 3.9 or newer. It depends on JAX for array computation and JIT compilation, matplotlib for plotting, and flax/flaxmodels for the `DNN` step.

JAX can run on CPU. For larger fields and image-processing pipelines, use a CUDA-capable GPU with a compatible JAX installation.

## Install From A Local Checkout

```bash
pip install -e /path/to/juniper
```

Optional CUDA dependency groups are available:

```bash
pip install -e "/path/to/juniper[cuda12]"
pip install -e "/path/to/juniper[cuda13]"
```

Use the CUDA extra that matches the JAX version and NVIDIA driver available on your machine.

## Verify The Installation

```bash
python -c "import juniper; print('JUNIPER import successful')"
```

Check JAX devices:

```python
import jax
print(jax.devices())
```

If no GPU is listed, JUNIPER still works on CPU. GPU availability is controlled by the installed JAX build and local driver setup.
