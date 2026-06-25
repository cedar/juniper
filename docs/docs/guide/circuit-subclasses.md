# Circuit Subclasses

For reusable architecture components, subclass `Circuit` and build the internal graph inside `__init__`. Instances of the subclass can then be imported, initialized, connected, compiled, and recorded like normal steps because a `Circuit` is also a connectable element.

## Define A Reusable Circuit

```python title="circuits.py"
from juniper import Circuit, StaticGain, Sum


class WeightedSum(Circuit):
    def __init__(self, name: str, left_gain: float = 1.0, right_gain: float = 1.0):
        super().__init__(name)

        with self as circuit:
            circuit.register_input_slot("left")
            circuit.register_input_slot("right")
            circuit.register_output_slot("out0")

            left = StaticGain("left_gain", factor=left_gain)
            right = StaticGain("right_gain", factor=right_gain)
            total = Sum("total")

            circuit.left >> left >> total
            circuit.right >> right >> total
            total >> circuit.out0
```

The `with self as circuit` block temporarily makes the subclass instance the current parent circuit. Steps created inside the block become internal elements of that circuit. When the block exits, the circuit structure is finalized and the subclass instance is registered in the surrounding parent circuit.

## Use It Like A Step

Import and instantiate the circuit class where you build an architecture. Connect to named slots when the circuit has multiple inputs.

```python title="architecture.py"
from juniper import CustomInput, StaticGain, get_arch
from circuits import WeightedSum


arch = get_arch()

left_input = CustomInput("left_input", shape=(1,))
right_input = CustomInput("right_input", shape=(1,))
mixer = WeightedSum("mixer", left_gain=0.5, right_gain=2.0)
output_gain = StaticGain("output_gain", factor=1.0)

left_input >> mixer.left
right_input >> mixer.right
mixer >> output_gain

arch.compile(warmup=1)
recording, timing = arch.run_simulation(
    num_steps=10,
    steps_to_record=[mixer, "mixer.total", output_gain],
)
```

The subclass instance has normal circuit paths. In this example, the public circuit is `mixer`, and internal elements can be addressed as `mixer.left_gain`, `mixer.right_gain`, and `mixer.total`.

## Design Notes

- Call `super().__init__(name)` before registering slots or creating internal elements.
- Register input and output slots on the circuit instance, not on internal steps.
- Use unique internal element names; they only need to be unique within the subclass instance.
- Use named input slots for multi-input components so callers can connect unambiguously.
- Connect one or more internal outputs to the circuit output slots.
- Record either the whole circuit instance, a public slot, or an internal path such as `"mixer.total"`.
