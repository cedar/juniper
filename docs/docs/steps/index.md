# Steps Reference

Steps are graph elements that execute JAX-compatible kernels. Most steps have default slots `in0` and `out0`; specialized steps add extra slots, outputs, or buffers.

## Common Rules

- Default input and output slots are named `in0` and `out0`.
- Slots are available as attributes, such as `step.in0`, `step.out0`, or `step.in1`.
- Incoming values are summed unless a step defines another aggregation rule.
- Sources have no input slots and write data before each tick.
- Sinks read data after each tick.
- Dynamic steps maintain buffers across ticks.
- Shapes and dtypes are inferred during compilation.

## Categories

| Category | Steps |
|----------|-------|
| [Algebra](algebra/index.md) | `AddConstant`, `ComponentMultiply`, `Convolution`, `Normalization`, `StaticGain`, `Sum`, `TransferFunction` |
| [Arrays](arrays/index.md) | `Clamp`, `CompressAxes`, `ExpandAxes`, `Flip`, `MatrixPadding`, `MatrixSlice`, `Projection`, `ReorderAxes`, `Resize`, `ScalarsToVector`, `VectorToScalars` |
| [Dynamic Field Theory](dft/index.md) | `NeuralField`, `SpaceToRateCode`, `RateToSpaceCode`, `HebbianConnection`, `BCMConnection` |
| [Image Processing](image_processing/index.md) | `ColorConversion`, `RGB2HSV`, `ColorFMap`, `RemoveBlackWhiteGreys`, `ShuffleImage`, `ViewportCamera`, `DNN` |
| [Robotics](robotics/index.md) | `CoordinateTransformation`, `FieldToPointCloud`, `PointCloudToField`, `PinHoleProjector`, `PinHoleBackProjector`, `PointCloudToRangeImage`, `RangeImageToPointCloud` |
| [Sources](sources/index.md) | `CustomInput`, `GaussInput`, `DemoInput`, `ImageLoader`, `TCPReader`, `TimedBoost` |
| [Sinks](sinks/index.md) | `StaticDebug`, `TCPWriter` |
