# Configurables

Configurables are helper objects used when constructing steps. They are not graph elements and do not run as separate nodes during simulation.

Unlike most steps, configurables take parameter dictionaries.

| Configurable | Purpose |
|--------------|---------|
| `Gaussian` | Builds Gaussian kernels for inputs, convolutions, and neural fields. |
| `LateralKernel` | Combines multiple kernel objects into one lateral interaction kernel. |
| `Sigmoid` | Selects one of JUNIPER's built-in transfer functions. |
| `Transform` | Wraps a function that returns a 4x4 coordinate transform matrix. |
| `FrameGraph` | Stores named coordinate frames and transform edges. |
