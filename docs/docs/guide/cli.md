# Command-Line Reference

JUNIPER is run via `run.py`, which takes an architecture file and simulates it.

```bash
python run.py <architecture_file> [options]
```

## Mandatory Argument

| Argument | Description |
|----------|-------------|
| `arch` | Path to the architecture file. Can be a `.py` file (implementing `get_architecture(args)`) or a JSON file (e.g., exported from CEDAR). |

## Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--cpu` | Force JAX to use CPU instead of GPU. | GPU if available |
| `--num_ticks N` | Number of simulation time steps. | - |
| `--num_runs N` | Number of full simulation runs. Steps are reset between runs. Useful for benchmarking. | 1 |
| `--recording name [name ...]` | Record buffers for plotting. Accepts step names (`field1`), slot names (`field1.out0`), or internal buffer names (`field1.activation`). Automatically shows a plot after simulation. | - |
| `--save_plot` | Save the recording plot to `output/plot_<timestamp>.png` instead of displaying it. Only effective with `--recording`. | Off |
| `--cache_jitted_funcs` | Cache JIT-compiled functions to disk. Reduces compile time on subsequent runs of the same architecture at the cost of disk space. | Off |
| `--static_euler_compilation` | Pre-compile each NeuralField's euler function individually with fixed parameters. Improves simulation performance at the cost of increased compilation time. Best for tuned architectures running many time steps. | Off |
| `--arch_args val [val ...]` | Pass arguments to the architecture's `get_architecture(args)` function as a list of strings. | `[]` |

## Examples

```bash
# Basic run
python run.py architectures/my_arch.py

# Run 500 ticks on CPU, record two fields
python run.py my_arch.py --cpu --num_ticks 500 --recording field1 field2

# Save plot and use cached compilation
python run.py my_arch.py --num_ticks 1000 --recording field1 --save_plot --cache_jitted_funcs

# Pass arguments to the architecture script
python run.py my_arch.py --arch_args 3 5x5 1x1

# Optimized long run
python run.py my_arch.py --num_ticks 10000 --static_euler_compilation
```
