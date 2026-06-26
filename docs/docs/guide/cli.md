# Command-Line Reference

`run.py` executes a Python architecture file from the command line.

```bash
python run.py <architecture_file> [options]
```

The architecture file must define `get_architecture()` or `get_architecture(args)`. The function should build the graph and return the top-level architecture. Returning `None` is also accepted; in that case the current `get_arch()` architecture is used.

## Example Architecture File

```python title="my_arch.py"
import numpy as np
import juniper as jp


def get_architecture(args=None):
    arch = jp.get_arch("cli_demo")

    source = jp.CustomInput("source", shape=(1,))
    source.set_data(np.array([1.0], dtype=np.float32))
    gain = jp.StaticGain("gain", factor=2.0)
    source >> gain

    return arch
```

Run it:

```bash
python run.py my_arch.py --num_ticks 100 --recording gain
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `arch` | Python file containing `get_architecture()`. | Required |
| `--cpu` | Force JAX to use CPU. | Off |
| `--num_ticks N` | Number of ticks per run. | `10` |
| `--num_runs N` | Number of simulation runs. The state is reset between runs. | `1` |
| `--warmup N` | Warmup ticks during compilation. | `3` |
| `--recording name [name ...]` | Recording targets, such as `field`, `field.out0`, or `field.activation`. | None |
| `--save_plot` | Save the recording plot to `output/recording_plot.png`. | Off |
| `--cache_jitted_funcs` | Enable JAX persistent compilation cache in the repository cache folder. | Off |
| `--static_euler_compilation` | Precompile individual neural-field Euler functions. | Off |
| `--arch_args val [val ...]` | Strings passed to `get_architecture(args)`. | `[]` |

## Common Commands

```bash
python run.py my_arch.py
python run.py my_arch.py --cpu --num_ticks 500
python run.py my_arch.py --num_ticks 1000 --recording field field.activation
python run.py my_arch.py --recording field --save_plot
python run.py my_arch.py --arch_args 3 50x50
```
