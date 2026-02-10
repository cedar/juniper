# Steps Reference

Steps are the fundamental building blocks of JUNIPER architectures. Each step reads from its input slots, performs a computation, and writes to its output slots.

## Step Basics

Every step inherits from the `Step` base class and implements a `compute()` method. Steps are created by providing a unique name and a parameter dictionary:

```python
step = StepClass("unique_name", {"param1": value1, "param2": value2})
```

### Input/Output Slots

- By default, every step has one input slot (`in0`) and one output slot (`out0`).
- Some steps register additional slots (e.g., `ColorConversion` has `out0`, `out1`, `out2` for H, S, V channels).
- Slots are accessed via `step.i0`, `step.i1`, ... (inputs) and `step.o0`, `step.o1`, ... (outputs).

### Static vs. Dynamic

- **Static steps** perform feed-forward computation with no internal state. They are executed once per tick in topological order.
- **Dynamic steps** maintain evolving state (e.g., activation in a neural field) and are updated via Euler integration each tick. Dynamic steps require a `shape` parameter.

## Categories

| Category | Description | Steps |
|----------|-------------|-------|
| [Algebra](algebra/index.md) | Mathematical operations on matrices | AddConstant, ComponentMultiply, Convolution, Normalization, StaticGain, Sum, TransferFunction |
| [Arrays](arrays/index.md) | Array shape manipulation and transformation | Clamp, CompressAxes, ExpandAxes, Flip, MatrixPadding, MatrixSlice, Projection, ReorderAxes, Resize, ScalarsToVector, VectorToScalars |
| [Dynamic Field Theory](dft/index.md) | Neural field dynamics and coding | NeuralField, HebbianConnection, RateToSpaceCode, SpaceToRateCode |
| [Image Processing](image_processing/index.md) | Image and vision processing | ColorConversion, DNN |
| [Robotics](robotics/index.md) | Coordinate transforms and spatial representations | CoordinateTransformation, FieldToVectors, PinHoleBackProjector, RangeImageToVectors, VectorsToField, VectorsToRangeImage |
| [Sources](sources/index.md) | Input data generation | CustomInput, DemoInput, GaussInput, HSV_input, ImageLoader, TCPReader, TimedBoost |
| [Sinks](sinks/index.md) | Output data transmission | TCPWriter |
