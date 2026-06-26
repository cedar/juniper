# Changelog

## Version 0.9.0

### Architecture Construction

- Most step constructors use explicit arguments, for example `StaticGain("gain", factor=0.8)`.
- `Element.from_params(name, params)` and Python dictionary unpacking remain available for code that stores parameters in dictionaries.
- Configurable helpers such as `Gaussian`, `LateralKernel`, `FrameGraph`, and `Transform` use parameter dictionaries.
- `get_arch()` returns the singleton top-level architecture. Use `delete_arch()` before building a separate architecture in the same Python process.

### Circuits And Connections

- `Circuit` can be used as a context manager for reusable nested graphs.
- Connections support element objects, slot objects, and string paths.
- Nested paths use dot notation, such as `controller.field.activation`.
- Connection validation reports duplicate connections, missing endpoints, and exceeded fan-in limits.

### Compilation And Simulation

- Architectures are compiled with `arch.compile(...)` and simulated with `arch.run_simulation(...)`.
- `run_simulation` returns `(Recording, TimingInfo)`.
- `arch.reset_state()` restores the post-compilation initial state for repeated runs.
- `arch.close_connections()` closes runtime I/O endpoints such as TCP workers.

### Recording And Plotting

- Recording targets can be strings, elements, slots, or buffers.
- `Recording` supports filtering by target, filtering by time interval, slicing, plotting, appending, saving, and loading.
- Saved recordings use a `manifest.json` file and one pickle file per recorded time step.

### Step Library

- Algebra, array, DFT, image-processing, robotics, source, and sink steps are documented in the step reference.
- Robotics vector/range-image terminology uses point-cloud names: `PointCloudToField`, `FieldToPointCloud`, `PointCloudToRangeImage`, and `RangeImageToPointCloud`.

## Migration Notes

### Constructor Dictionaries

Replace step parameter dictionaries with explicit constructor arguments:

```python
# Before
gain = StaticGain("gain", {"factor": 0.8})

# After
gain = StaticGain("gain", factor=0.8)
```

For stored parameter dictionaries:

```python
field = NeuralField.from_params("field", params)
# or
field = NeuralField("field", **params)
```

### Simulation Loops

Use `run_simulation` for fixed-step execution:

```python
arch.compile(warmup=1)
recording, timing = arch.run_simulation(
    num_steps=100,
    steps_to_record=["field", "field.activation"],
)
```

### CLI Architecture Files

Architecture files used with `run.py` should define `get_architecture()` or `get_architecture(args)`:

```python
import juniper as jp


def get_architecture(args=None):
    arch = jp.get_arch("demo")
    source = jp.CustomInput("source", shape=(1,))
    gain = jp.StaticGain("gain", factor=2.0)
    source >> gain
    return arch
```
