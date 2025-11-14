from juniper.sources import GaussInput
from juniper.statics import Normalization
from juniper.statics import Projection
from juniper.statics import Sum
from juniper.statics import TransferFunction
from juniper.dynamics import NeuralField
from juniper.configurables import Gaussian
from juniper.statics import ComponentMultiply

from juniper.robotics import SpaceToRateCode
from juniper.robotics import RateToSpaceCode

from juniper.statics import CompressAxes

def get_architecture(args):
    shape1 = (50,)
    shape2 = (50,50,25)


    nf1 = NeuralField(f"nf1", {"shape": shape1, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1,
                        "LateralKernel": Gaussian({"sigma": (3,), "amplitude": 1, "normalized": True, "max_shape": shape1}),})
                        
    nf2 = NeuralField(f"nf2", {"shape": (50,50), "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1,
                        "LateralKernel": Gaussian({"sigma": (3,3), "amplitude": 1, "normalized": True, "max_shape": (50,50)}),})
    
    nf3 = NeuralField(f"nf3", {"shape": shape2, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1,})

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

    s2r = SpaceToRateCode('s2r', params={"shape": shape2, "limits": tuple([(0,shape2[i]) for i in range(len(shape2))])})
    r2s = RateToSpaceCode('r2s', params={'shape': shape2, "limits": tuple([(0,shape2[i]) for i in range(len(shape2))]), "amplitude":2, "sigma": (2,2,2)})
    comp1 = CompressAxes('comp1', params={"axis": (2,), "compression_type": "Maximum"})
    comp2 = CompressAxes('comp2', params={"axis": (2,), "compression_type": "Maximum"})

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

    gi1 >> s2r >> r2s >> nf3
    gi1 >> comp1 >> nf2
    r2s >> comp2 >> nf2
