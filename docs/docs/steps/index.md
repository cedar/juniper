# Steps Reference

Steps are graph elements that read input slots, execute a JAX-compatible compute kernel, and write output slots. Most steps have default `in0` and `out0` slots; specialized steps register extra inputs, outputs, or buffers.

## Constructor Style

The current API uses explicit constructor arguments for steps:

```python
from juniper import StaticGain, Sum

gain = StaticGain("gain", factor=2.0)
add = Sum("add")
```

Use `Element.from_params(name, params)` if you need to instantiate from a dictionary.

## Slot And State Rules

- Default input/output slots are named `in0` and `out0`.
- Slots are available as attributes, such as `step.in0`, `step.out0`, `step.in1`, or `step.learn_node`.
- Multiple inputs are summed unless the step changes `input_aggregation`, as `ComponentMultiply` does.
- Sources and sinks are runtime I/O endpoints; dynamic steps maintain buffers across ticks.
- Shape and dtype inference happen at compile time. Steps with nontrivial output shape override `infer_output_shapes`.

## Categories

| Category | Steps |
|----------|-------|
| [Algebra](algebra/index.md) | `AddConstant`, `ComponentMultiply`, `Convolution`, `Normalization`, `StaticGain`, `Sum`, `TransferFunction` |
| [Arrays](arrays/index.md) | `Clamp`, `CompressAxes`, `ExpandAxes`, `Flip`, `MatrixPadding`, `MatrixSlice`, `Projection`, `ReorderAxes`, `Resize`, `ScalarsToVector`, `VectorToScalars` |
| [Dynamic Field Theory](dft/index.md) | `BCMConnection`, `HebbianConnection`, `NeuralField`, `RateToSpaceCode`, `SpaceToRateCode` |
| [Image Processing](image_processing/index.md) | `ColorConversion`, `ColorFMap`, `DNN`, `RGB2HSV`, `RemoveBlackWhiteGreys`, `ShuffleImage`, `ViewportCamera` |
| [Robotics](robotics/index.md) | `CoordinateTransformation`, `FieldToPointCloud`, `PinHoleBackProjector`, `PinHoleProjector`, `PointCloudToField`, `PointCloudToRangeImage`, `RangeImageToPointCloud` |
| [Sources](sources/index.md) | `CustomInput`, `DemoInput`, `GaussInput`, `ImageLoader`, `TCPReader`, `TimedBoost` |
| [Sinks](sinks/index.md) | `StaticDebug`, `TCPWriter` |
