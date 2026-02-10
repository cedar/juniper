# Building Architectures

An **architecture** in JUNIPER is a directed computational graph of **steps** connected through **slots**. Each step performs a computation on its inputs and writes the result to its outputs. JUNIPER distinguishes between **static steps** (feed-forward, computed once per tick) and **dynamic steps** (time-evolving, e.g. neural fields).

## Creating an Architecture

Architecture files are Python scripts that define a `get_architecture(args)` function. This function creates steps, connects them, and the architecture is returned implicitly via the global singleton.

```python
from juniper import GaussInput, StaticGain, NeuralField, Gaussian
from juniper.Architecture import get_arch

def get_architecture(args):
    # 1. Instantiate steps
    gi = GaussInput("input", {"shape": (50,), "sigma": (3,), "amplitude": 5})
    gain = StaticGain("gain", {"factor": 0.8})
    nf = NeuralField("field", {
        "shape": (50,),
        "resting_level": -5,
        "global_inhibition": -0.01,
        "tau": 0.1,
        "input_noise_gain": 0.1,
        "sigmoid": "AbsSigmoid",
        "beta": 100,
        "theta": 0.5,
        "LateralKernel": Gaussian({
            "sigma": (3,),
            "amplitude": 5,
            "normalized": True,
            "max_shape": (50,),
        }),
    })

    # 2. Connect steps
    gi >> gain >> nf

    # 3. Return the architecture
    return get_arch()
```

## Instantiating Steps

Every step takes two arguments:

1. **`name`** (`str`) -- A unique identifier. Must not contain dots.
2. **`params`** (`dict`) -- A dictionary of parameters specific to that step.

```python
gain = StaticGain("my_gain", {"factor": 2.5})
```

Steps are automatically registered in the global architecture singleton upon creation.

## Connecting Steps

JUNIPER provides several equivalent ways to connect steps. All create a directed edge from an output slot to an input slot.

### The `>>` Operator (Recommended)

The right-shift operator connects the default output of the left step to the default input of the right step. It returns the right-hand operand, allowing chaining.

```python
# Single connection
step_a >> step_b

# Chained connections: A -> B -> C -> D
step_a >> step_b >> step_c >> step_d
```

### The `<<` Operator

The left-shift operator connects in reverse -- the right step's output flows into the left step's input.

```python
# step_b receives output from step_a
step_b << step_a
```

### Named Slots

Steps can have multiple input and output slots. Use the slot accessors `o0`, `o1`, ... (outputs) and `i0`, `i1`, ... (inputs) to target specific slots.

```python
# Connect output slot 1 of step_a to input slot 2 of step_b
step_a.o1 >> step_b.i2
```

You can also use string identifiers with the `>>` operator:

```python
step_a >> "step_b.in1"
```

### Explicit `connect_to`

The Architecture class provides a direct method:

```python
from juniper.Architecture import get_arch
arch = get_arch()
arch.connect_to("step_a.out0", "step_b.in0")
```

### Multiple Inputs

Some steps accept multiple incoming connections on the same input slot (e.g., `Sum`, `NeuralField`). When multiple connections arrive at the same slot, their values are **summed** automatically.

```python
step_a >> nf
step_b >> nf
step_c >> nf  # All three are summed into nf's input
```

## Static vs. Dynamic Steps

| | Static Steps | Dynamic Steps |
|---|---|---|
| **Behavior** | Feed-forward computation | Time-evolving state (e.g., differential equations) |
| **Examples** | `StaticGain`, `Sum`, `Convolution` | `NeuralField`, `HebbianConnection`, `SpaceToRateCode` |
| **`is_dynamic`** | `False` (default) | `True` |
| **Requires `shape`** | No | Yes |
| **Execution** | Computed in topological order before dynamic steps | Updated via euler integration each tick |

## Compiling and Running

After all steps are created and connected, the architecture must be **compiled** before it can be run.

```python
arch = get_arch()

# Compile (includes warmup JIT pass)
arch.compile()

# Run the simulation tick by tick
for _ in range(1000):
    arch.tick()
```

### Using `run_simulation`

For convenience, use `run_simulation` which handles the loop, timing, and buffer recording:

```python
history, ms_per_tick, timing = arch.run_simulation(
    tick_func=arch.tick,
    steps_to_plot=["field.out0"],
    num_steps=1000,
)
```

## Running from the Command Line

```bash
python run.py my_architecture.py --num_ticks 500 --recording field
```

See the [Command-Line Reference](cli.md) for all options.

## Complete Example

```python
from juniper import GaussInput, NeuralField, Gaussian, Normalization, Sum
from juniper.Architecture import get_arch

def get_architecture(args):
    shape = (50,)

    # Sources
    gi1 = GaussInput("gi1", {"shape": shape, "sigma": (2,), "amplitude": 3})
    gi2 = GaussInput("gi2", {"shape": shape, "sigma": (4,), "amplitude": 1,
                              "center": (10,)})

    # Processing
    norm = Normalization("norm", {"function": "L2Norm"})
    add  = Sum("add", {})

    # Neural field
    nf = NeuralField("nf", {
        "shape": shape,
        "resting_level": -5,
        "global_inhibition": -0.01,
        "tau": 0.1,
        "input_noise_gain": 0.1,
        "sigmoid": "AbsSigmoid",
        "beta": 100,
        "theta": 0.5,
        "LateralKernel": Gaussian({
            "sigma": (3,),
            "amplitude": 5,
            "normalized": True,
            "max_shape": shape,
        }),
    })

    # Wire it up
    gi1 >> norm >> add
    gi2 >> add
    add >> nf

    return get_arch()
```
