from juniper.steps.GaussInput import GaussInput
from juniper.steps.NeuralField import NeuralField
from juniper.steps.StaticGain import StaticGain
from juniper.Gaussian import Gaussian

# Supply args for this architecture using the --arch_args command line argument in the following format:
# num_fields shape kernel_sigma_shape gauss_mode
# e.g. '3 50x50 3x1 singlegauss' or '10 50x50x50 3x3x3 multigauss'
def get_architecture(args):
    if len(args) != 4:
        raise Exception("Expected exactly 4 arguments, got " + str(len(args)))
    
    # Parse arguments
    num_fields = int(args[0])
    shape = tuple([int(size) for size in args[1].split("x")])
    kernel_sigmas = tuple([float(size) for size in args[2].split("x")])
    gauss_input_sigma = tuple([float(size) for size in args[2].split("x")])
    if not args[3] in ["singlegauss", "multigauss"]:
        raise Exception("Expected 'singlegauss' or 'multigauss', got " + args[3])
    single_gauss = args[3] == "singlegauss"

    # Some static parameters
    kernel_amplitude = 1
    amplitude = 2
    factor = 2

    # Build architecture
    for i in range(num_fields):
        if i == 0 or not single_gauss:
            gi = GaussInput(f"gi{i}", {"shape": shape, "sigma": gauss_input_sigma, "amplitude": amplitude + i * 0.01})
            st = StaticGain(f"st{i}", {"factor": factor})
            gi >> st
        nf = NeuralField(f"nf{i}", {"resting_level": -0.7+i*0.001, "global_inhibition": -0.01+i*0.001, "tau": 0.1, 
                                    "sigmoid": "AbsSigmoid", "beta": 100+i*0.001, "theta": 0+i*0.001,
                            "input_noise_gain": 0.1+i*0.001, 
                            "lateral_kernel_convolution": 
                            Gaussian({"sigma": kernel_sigmas, "amplitude": kernel_amplitude, "normalized": True, "max_shape": shape}),
                            "shape": shape})
        from_element = f"st0" if single_gauss else f"st{i}"
        nf << from_element
