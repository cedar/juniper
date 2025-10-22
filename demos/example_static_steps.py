from juniper.steps.GaussInput import GaussInput
from juniper.steps.Normalization import Normalization
from juniper.steps.Projection import Projection
from juniper.steps.Sum import Sum
from juniper.steps.TransferFunction import TransferFunction
from juniper.steps.NeuralField import NeuralField
from juniper.Gaussian import Gaussian
from juniper.steps.ComponentMultiply import ComponentMultiply

def get_architecture(args):
    shape1 = (50,)
    shape2 = (50,50,25)


    nf1 = NeuralField(f"nf1", {"shape": shape1, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1,
                        "lateral_kernel_convolution": Gaussian({"sigma": (3,), "amplitude": 1, "normalized": True, "max_shape": shape1}),})
                        
    nf2 = NeuralField(f"nf2", {"shape": (50,50), "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1,
                        "lateral_kernel_convolution": Gaussian({"sigma": (3,3), "amplitude": 1, "normalized": True, "max_shape": (50,50)}),})

    # Static steps
    gi0 = GaussInput("gi0", {"shape": shape1, "sigma": (2,), "amplitude": 2})
    gi1 = GaussInput("gi1", {"shape": shape2, "sigma": (2,2,2), "amplitude": 2})

    projection1 = Projection("proj1", {"input_shape": shape2, "output_shape": shape1, "axis": (1,2), "order":(0,), "compression_type": "Sum"})
    projection0 = Projection("proj0", {"input_shape": shape1, "output_shape": (50,50), "axis": (0,), "order":(0,1), "compression_type": "Sum"})
    projection2 = Projection("proj2", {"input_shape": (50,50), "output_shape": (50,50), "axis": (0,1), "order":(1,0), "compression_type": "Sum"})

    norm0 = Normalization("norm0", {"function": "L2Norm"})
    norm1 = Normalization("norm1", {"function": "L2Norm"})

    trans0 = TransferFunction("trans0", {"function": "ExpSigmoid", "threshold": 0., "beta": 1.})

    sum0 = Sum("sum0", {})

    comp_multiply = ComponentMultiply("comp_multiply", {})

    # Connections (different syntax possible)
    gi0 >> norm0
    gi1 >> projection1
    projection1 >> trans0
    trans0 >> norm1

    norm0 >> sum0
    norm1 >> sum0

    sum0 >> comp_multiply >> nf1

    nf1 >> projection0

    projection0 >> projection2
    projection2 >> nf2

