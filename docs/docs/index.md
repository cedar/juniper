# JUNIPER


Welcome to JUNIPER, the GPU Accelerated Python Implementation of CEDAR.
JUNIPER is a simulation library for neural dynamic architectures. It combines a small frontend for connecting processing elements with a JAX backend that compiles the graph into simulation kernels.

The library is centered on Dynamic Field Theory, where continuous activation fields evolve over time under input, recurrent interaction, resting dynamics, and noise. JUNIPER also includes general array operations, algebraic transforms, image-processing steps, robotics coordinate transforms, TCP I/O, recording, plotting, and reusable nested circuits.

## Minimal Example

```python
import numpy as np
import juniper as jp

arch = jp.get_arch("example")

inp = jp.CustomInput("inp", shape=(50,))
inp.set_data(np.ones((50,), dtype=np.float32))

field = jp.NeuralField("field", shape=(50,), resting_level=-5.0)
inp >> field

arch.compile(warmup=1)
recording, timing = arch.run_simulation(
    num_steps=100,
    steps_to_record=["field", "field.activation"],
)
recording.plot(keys=["field.activation"])
```

## Recommended Reading Order

1. [Installation](guide/installation.md)
2. [Building Architectures](guide/architecture.md)
3. [Steps Reference](steps/index.md)
4. [Configurables](configurables/index.md)
5. [API Reference](api/index.md)
6. [Changelog](changelog.md)
