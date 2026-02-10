# Array Steps

Array steps handle reshaping, slicing, padding, and other structural transformations of arrays. These are all static steps.

| Step | Description |
|------|-------------|
| [Clamp](clamp.md) | Clamps values to a min/max range |
| [CompressAxes](compress_axes.md) | Reduces dimensions by aggregation (sum, average, max, min) |
| [ExpandAxes](expand_axes.md) | Adds and repeats new dimensions |
| [Flip](flip.md) | Reverses array along specified axes |
| [MatrixPadding](matrix_padding.md) | Pads a matrix with border elements |
| [MatrixSlice](matrix_slice.md) | Extracts a sub-region of a matrix |
| [Projection](projection.md) | Combined expand/compress and reorder operation |
| [ReorderAxes](reorder_axes.md) | Permutes array dimensions |
| [Resize](resize.md) | Resizes to a new shape with interpolation |
| [ScalarsToVector](scalars_to_vector.md) | Combines N scalar inputs into a vector |
| [VectorToScalars](vector_to_scalars.md) | Splits a vector into N scalar outputs |
