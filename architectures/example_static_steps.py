from src.steps.GaussInput import GaussInput
from src.steps.Normalization import Normalization
from src.steps.Projection import Projection
from src.steps.Sum import Sum
from src.Architecture import Architecture
from src.steps.TransferFunction import TransferFunction
from src.steps.NeuralField import NeuralField
from src.AbsSigmoid import AbsSigmoid
from src.GaussKernel import GaussKernel

def get_architecture(args):
    arch = Architecture()
    shape1 = (50,)
    shape2 = (50,50,25)


    nf1 = NeuralField(f"nf1", {"shape": shape1, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.1, 
                        "input_noise_gain": 0.1, "sigmoid": AbsSigmoid(100, 0),
                        "lateral_kernel_convolution": GaussKernel({"sigma": (3,), "amplitude": 1, "normalized": True, "max_shape": shape1}),})
                        
    nf2 = NeuralField(f"nf2", {"shape": (50,50), "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.1, 
                        "input_noise_gain": 0.1, "sigmoid": AbsSigmoid(100, 0),
                        "lateral_kernel_convolution": GaussKernel({"sigma": (3,3), "amplitude": 1, "normalized": True, "max_shape": (50,50)}),})

    # Static steps
    gi0 = GaussInput("gi0", {"shape": shape1, "sigma": (2,), "amplitude": 2})
    gi1 = GaussInput("gi1", {"shape": shape2, "sigma": (2,2,2), "amplitude": 2})

    projection1 = Projection("proj1", {"input_shape": shape2, "output_shape": shape1, "mapping": (1,), "compression_type": "Sum"})
    projection0 = Projection("proj0", {"input_shape": shape1, "output_shape": (50,50), "mapping": (1,0), "compression_type": "Sum"})
    projection2 = Projection("proj2", {"input_shape": (50,50), "output_shape": (50,50), "mapping": (1,0), "compression_type": "Sum"})

    norm0 = Normalization("norm0", {"function": "L2Norm"})
    norm1 = Normalization("norm1", {"function": "L2Norm"})

    trans0 = TransferFunction("trans0", {"function": "ExpSigmoid", "threshold": 0., "beta": 1.})

    sum0 = Sum("sum0", {})

    # Add steps to architecture
    arch += gi0
    arch += gi1

    arch += nf1

    # Dynamic steps
    arch += projection1
    arch += projection0
    arch += norm0
    arch += norm1
    arch += trans0
    arch += sum0

    arch += nf2
    arch += projection2

    # Connections (different syntax possible)
    gi0 >> norm0
    gi1 >> projection1
    projection1 >> trans0
    trans0 >> norm1

    norm0 >> sum0
    norm1 >> sum0

    sum0 >> nf1

    nf1 >> projection0

    projection0 >> projection2
    projection2 >> nf2

    return arch
