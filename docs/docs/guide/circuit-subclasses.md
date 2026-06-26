# Circuit Subclasses

Subclass `Circuit` when you want to package a group of connected elements as a reusable component. A circuit instance can be connected, compiled, and recorded like any other element.

## Define A Circuit

```python title="circuits.py"
import juniper as jp


class WeightedSum(jp.Circuit):
    def __init__(self, name: str, left_gain: float = 1.0, right_gain: float = 1.0):
        super().__init__(name)

        with self as circuit:
            circuit.register_input_slot("left")
            circuit.register_input_slot("right")
            circuit.register_output_slot("out0")

            left = jp.StaticGain("left_gain", factor=left_gain)
            right = jp.StaticGain("right_gain", factor=right_gain)
            total = jp.Sum("total")

            circuit.left >> left >> total
            circuit.right >> right >> total
            total >> circuit.out0
```

`with self as circuit` makes the instance the active parent while the internal graph is created. When the block exits, the circuit is finalized and registered in the surrounding parent circuit.

## Use The Circuit

```python title="architecture.py"
import numpy as np
import juniper as jp
from circuits import WeightedSum


def get_architecture():
    arch = jp.get_arch("weighted_sum_demo")

    left = jp.CustomInput("left", shape=(1,))
    right = jp.CustomInput("right", shape=(1,))
    left.set_data(np.array([1.0], dtype=np.float32))
    right.set_data(np.array([2.0], dtype=np.float32))

    mixer = WeightedSum("mixer", left_gain=0.5, right_gain=2.0)
    left >> mixer.left
    right >> mixer.right

    return arch
```

Record the public circuit, a public slot, or an internal path:

```python
recording, _ = arch.run_simulation(
    num_steps=10,
    steps_to_record=["mixer", "mixer.total", "mixer.left_gain.out0"],
)
```

## Guidelines

- Call `super().__init__(name)` before registering slots or creating internal elements.
- Register public inputs and outputs on the circuit instance.
- Use unique internal element names within the circuit.
- Prefer named input slots when the circuit has more than one input.
- Connect at least one internal source to each public output slot that should produce data.
