# Source Steps

Source steps generate input data for the architecture. They have no input connections and produce output that feeds into other steps.

| Step | Description |
|------|-------------|
| [CustomInput](custom_input.md) | Programmatically settable input |
| [DemoInput](demo_input.md) | Gaussian input with runtime-adjustable parameters |
| [GaussInput](gauss_input.md) | Static Gaussian bump input |
| [HSV_input](hsv_input.md) | Converts RGB input to HSV channels |
| [ImageLoader](image_loader.md) | Loads an image file from disk |
| [TCPReader](tcp_reader.md) | Receives matrix data over TCP |
| [TimedBoost](timed_boost.md) | Outputs a scalar boost during a configurable time window |
