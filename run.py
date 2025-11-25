import argparse
import jax
import time
import os

from juniper.util.plotting import plot_history
from juniper.util.util import tprint
from juniper import util
from juniper.Architecture import get_arch

if __name__ == "__main__":

    ## --- Argparse ---

    parser = argparse.ArgumentParser(description='JUNIPER: GPU Accelerated Python Implementation of CEDAR Using JAX')
    parser.add_argument('arch', type=str, help='Load architecture from JSON or Python file (i.e. JSON exported by CEDAR' + \
                        ' or python file containing get_architecture() function)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--recording', default=[], nargs="+", help='Activates recording mode. Specify steps or their buffers/output_slots to plot, e.g., \'step1 step2.buffer\' or \'"Neural Field1" "Static Gain1" "Neural Field2.activation"\' (without single quotes)')
    parser.add_argument('--num_ticks', type=int, default=10, help='Run simulation for n ticks')
    parser.add_argument('--num_runs', type=int, default=2, help='Run n simulations in total')
    parser.add_argument('--save_plot', action='store_true', help="Save plots to 'output/plot_<timestamp>.png' instead of displaying them")
    parser.add_argument('--cache_jitted_funcs', action='store_true', help="Persistently save jit-compiled functions to reduce compilation time when running the same architecture multiple times")
    parser.add_argument('--static_euler_compilation', action='store_true', help="Pre-compile the euler function of each individual field. Can improve performance, but increases compilation time.")
    parser.add_argument('--arch_args', default=[], nargs="+", help='Optional arguments for the architecture that is loaded')

    args = parser.parse_args()

    ## --- Initialization ---

    if args.cpu:
        jax.config.update('jax_platform_name', 'cpu')
    if args.cache_jitted_funcs:
        jax.config.update("jax_compilation_cache_dir", os.path.join(util.root(), "jax_cache"))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    print("Computing devices found by JAX:")
    print(jax.local_devices())
    if args.cpu:
        if not "Cpu" in str(jax.local_devices()):
            raise Exception("CPU not loaded. Make sure util_jax is not (in)directly imported before this line")

    # These imports have to happen *after* the jax config is set
    from juniper.util import util_jax
    import juniper.util.architecture_import as architecture_import

    util_jax.get_config()["euler_step_static_precompile"] = args.static_euler_compilation == True
    util_jax.get_config()["arch_file_path"] = args.arch

    ## --- Load architecture ---
    arch = get_arch()
    architecture_import.import_file(args.arch, args.arch_args)
    tprint("Architecture loaded")

    ## --- Compile architecture ---

    compile_time = time.time()

    arch.compile()
    tprint("Architecture compiled")

    compile_time = time.time() - compile_time

    ## --- Simulation ---
    
    for i in range(args.num_runs): # Do multiple runs to check stability of timing results
        print(f"\nRun {i+1}")
        plot_data_history, ms_per_tick, timing = arch.run_simulation(arch.tick, args.recording, args.num_ticks)
        arch.reset_steps()

    print()
    tprint(f"Simulations done")

    ## --- Plotting ---
    if len(args.recording) > 0:
        plot_history(args.num_ticks, plot_data_history, args.save_plot, args.recording)


# TODO
# check TODOs in source code
# Add documentation to each class 
# Reorganize steps into categories dynamic, processing, sinks, sources, robotics, auxillaries
# use needs_incoming_connections to check if income slots needs to be specified and deal with bug when it is not.

# performance steps to update
"""
- Custom Input
- all other 
"""