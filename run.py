import argparse
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import jax

from juniper import get_arch
from juniper.util import util


def _load_architecture_module(path: str):
    arch_path = Path(path).resolve()
    if not arch_path.exists():
        raise FileNotFoundError(f"Architecture file does not exist: {arch_path}")
    if arch_path.suffix != ".py":
        raise ValueError("run.py currently supports Python architecture files with a get_architecture function.")

    sys.path.insert(0, str(arch_path.parent))
    module_name = f"juniper_user_arch_{arch_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, arch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load architecture module from {arch_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_architecture(module, arch_args):
    if not hasattr(module, "get_architecture"):
        raise AttributeError("Architecture file must define get_architecture(args) or get_architecture().")

    get_architecture = module.get_architecture
    signature = inspect.signature(get_architecture)
    if len(signature.parameters) == 0:
        arch = get_architecture()
    else:
        arch = get_architecture(arch_args)

    return arch if arch is not None else get_arch()


def _plot_recording(recording, args):
    if not args.recording:
        return

    snapshot_indices = None
    if args.num_ticks > 0:
        snapshot_indices = [0, args.num_ticks - 1]

    fig = recording.plot(
        keys=args.recording,
        idx_interval=None,
        time_axis=None,
        snapshot_indices=snapshot_indices,
        group_keys=None,
        figsize=(10, 4),
    )

    if args.save_plot:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "recording_plot.png"
        fig.savefig(output_path)
        print(f"Saved recording plot to {output_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="JUNIPER: GPU Accelerated Python Implementation of CEDAR Using JAX")
    parser.add_argument("arch", type=str, help="Python file containing get_architecture(args) or get_architecture()")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--recording",
        default=[],
        nargs="+",
        help="Record steps, slots, or buffers, e.g. field field.out0 field.activation",
    )
    parser.add_argument("--num_ticks", type=int, default=10, help="Run simulation for n ticks")
    parser.add_argument("--num_runs", type=int, default=1, help="Run n simulations in total")
    parser.add_argument("--warmup", type=int, default=3, help="Run n warmup ticks during compilation")
    parser.add_argument("--save_plot", action="store_true", help="Save the recording plot to output/recording_plot.png")
    parser.add_argument(
        "--cache_jitted_funcs",
        action="store_true",
        help="Persistently cache JIT-compiled functions for repeated runs of the same architecture",
    )
    parser.add_argument(
        "--static_euler_compilation",
        action="store_true",
        help="Pre-compile the euler function of each individual field",
    )
    parser.add_argument("--arch_args", default=[], nargs="+", help="Optional arguments passed to get_architecture(args)")

    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")
    if args.cache_jitted_funcs:
        jax.config.update("jax_compilation_cache_dir", os.path.join(util.root(), "jax_cache"))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    print("Computing devices found by JAX:")
    print(jax.local_devices())

    # These imports have to happen after the JAX config is set.
    from juniper.util import util_jax

    util_jax.get_config()["euler_step_static_precompile"] = args.static_euler_compilation
    util_jax.get_config()["arch_file_path"] = args.arch

    module = _load_architecture_module(args.arch)
    get_arch()
    arch = _build_architecture(module, args.arch_args)

    arch.compile(print_compile_info=True, warmup=args.warmup, load_buffer=False)

    recording = None
    for run_idx in range(args.num_runs):
        print(f"\nRun {run_idx + 1}")
        recording, _timing = arch.run_simulation(
            num_steps=args.num_ticks,
            steps_to_record=args.recording,
            print_timing=True,
        )
        if run_idx < args.num_runs - 1:
            arch.reset_state()

    if recording is not None:
        _plot_recording(recording, args)

    arch.close_connections()


if __name__ == "__main__":
    main()
