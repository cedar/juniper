# Configurables

Configurables are parameterizable helper objects used by steps. They do not participate in graph execution by themselves, but they provide kernels, nonlinearities, or transformation logic during step construction. Unlike most steps, configurables currently take parameter dictionaries.

| Configurable | Reference |
|--------------|-----------|
| `Gaussian` | [Gaussian](gaussian.md) |
| `LateralKernel` | [LateralKernel](lateral_kernel.md) |
| `Sigmoid` | [Sigmoid](sigmoid.md) |
| `FrameGraph` | [FrameGraph](frame_graph.md) |
| `Transform` | [Transform](transform.md) |
