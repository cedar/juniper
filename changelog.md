
# Migration Guide: Old Architectures To The New API

The refactor changes how architectures are built, run, recorded, and imported. Most old architectures can be migrated mechanically by updating constructors, returning the current architecture, and replacing old runtime calls with `run_simulation`.

### 1. Replace Parameter Dictionaries With Explicit Step Arguments

Most steps no longer take a single `params` dictionary. Pass parameters directly to the constructor.

Old:

```python
gain = StaticGain("gain", {"factor": 0.8})
field = NeuralField("field", {
    "shape": (50,),
    "resting_level": -5,
    "global_inhibition": -0.01,
    "tau": 0.1,
})
```

New:

```python
gain = StaticGain("gain", factor=0.8)
field = NeuralField(
    "field",
    shape=(50,),
    resting_level=-5,
    global_inhibition=-0.01,
    tau=0.1,
)
```

You can also use:

```python
gain = StaticGain("gain", factor=0.8)
field = NeuralField.from_params("field", {
    "shape": (50,),
    "resting_level": -5,
    "global_inhibition": -0.01,
    "tau": 0.1,
})
```

or simply unpack the dictionary:

```python
gain = StaticGain("gain", factor=0.8)
field = NeuralField("field", **{
    "shape": (50,),
    "resting_level": -5,
    "global_inhibition": -0.01,
    "tau": 0.1,
})
```



Configurables still use dictionaries:

```python
kernel = Gaussian({
    "shape": (50,),
    "sigma": (3,),
    "amplitude": 5.0,
    "normalized": True,
    "factorized": False,
})
field = NeuralField("field", shape=(50,), lateral_kernel=kernel)
```


### 2. Replace Tick Loops With `run_simulation`

The public `arch.tick()` workflow is gone. Compile once, then run a fixed number of ticks through `run_simulation`.

Old:

```python
arch.compile()
for _ in range(100):
    arch.tick()
```

New:

```python
arch.compile(warmup=1)
recording, timing = arch.run_simulation(
    num_steps=100,
    steps_to_record=["field", "field.activation"],
)
```

For repeated benchmark runs, call `arch.reset_state()` between runs.

### 3. Update Recording And Plotting Code

`run_simulation` returns a `Recording` object, not a raw history list.

```python
recording, timing = arch.run_simulation(
    num_steps=50,
    steps_to_record=["field.out0", "field.activation"],
)

field_output = recording.get_at_element("field.out0")
first_ten = recording.get_in_step_interval((0, 10))
recording.plot(keys=["field.out0"])
run_dir = recording.save_to_file("recordings")
```

Recording targets can be step names, slot names, buffer names, element objects, slot objects, or buffer objects.

### 4. Rename Removed Robotics And Source Classes as well as new parameter names

Some old documentation names were replaced by the current point-cloud/range-image naming:

| Old name | New name |
|----------|----------|
| `VectorsToField` | `PointCloudToField` |
| `FieldToVectors` | `FieldToPointCloud` |
| `VectorsToRangeImage` | `PointCloudToRangeImage` |
| `RangeImageToVectors` | `RangeImageToPointCloud` |
| `HSV_input` | Use current image/color processing steps such as `RGB2HSV`, `ColorConversion`, or `ColorFMap` |


Some parameter names, such as the shape parameter in the BCMConnection step have changed. 

### 9. Update CLI Usage

Architecture files should define `get_architecture(args)` or `get_architecture()`. Build steps as before, then return `get_arch()`.

```python
from juniper import CustomInput, StaticGain, get_arch


def get_architecture(args):
    source = CustomInput("source", shape=(1,))
    gain = StaticGain("gain", factor=2.0)

    source >> gain

    return get_arch()
```

`run.py` initializes the top-level architecture before calling `get_architecture`, so steps created inside the function register automatically.

`run.py` now loads Python architecture files with `get_architecture(args)` or `get_architecture()`.

```bash
python run.py my_arch.py --num_ticks 500 --recording field field.activation
python run.py my_arch.py --cpu --warmup 0
```

