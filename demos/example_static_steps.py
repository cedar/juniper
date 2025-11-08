from juniper.steps.GaussInput import GaussInput
from juniper.steps.Normalization import Normalization
from juniper.steps.Projection import Projection
from juniper.steps.Sum import Sum
from juniper.steps.TransferFunction import TransferFunction
from juniper.steps.NeuralField import NeuralField
from juniper.Gaussian import Gaussian
from juniper.steps.ComponentMultiply import ComponentMultiply

from juniper.steps.Resize import Resize
from juniper.steps.AddConstant import AddConstant
from juniper.steps.Clamp import Clamp
from juniper.steps.Flip import Flip
from juniper.steps.MatrixSlice import MatrixSlice
from juniper.steps.MatrixPadding import MatrixPadding

from juniper.steps.VectorToScalars import VectorToScalars
from juniper.steps.ScalarsToVector import ScalarsToVector

def get_architecture(args):
    shape1 = (50,)
    shape2 = (50,50,25)

    shape3 = (20,20)
    shape4 = (30,40)

    shape5 = (3,)
    shape6 = (1,)


    nf1 = NeuralField(f"nf1", {"shape": shape1, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1,
                        "LateralKernel": Gaussian({"sigma": (3,), "amplitude": 1, "normalized": True, "max_shape": shape1}),})
                        
    nf2 = NeuralField(f"nf2", {"shape": (50,50), "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1,
                        "LateralKernel": Gaussian({"sigma": (3,3), "amplitude": 1, "normalized": True, "max_shape": (50,50)}),})
    
    nf3 = NeuralField(f"nf3", {"shape": shape3, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1,
                        "LateralKernel": Gaussian({"sigma": (3,3), "amplitude": 1, "normalized": True, "max_shape": (50,50)}),})
    nf4 = NeuralField(f"nf4", {"shape": shape4, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1,
                        "LateralKernel": Gaussian({"sigma": (3,3), "amplitude": 1, "normalized": True, "max_shape": (50,50)}),})
    
    nf5 = NeuralField(f"nf5", {"shape": shape5, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1})
    nf6 = NeuralField(f"nf6", {"shape": shape5, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1})
    
    nf7 = NeuralField(f"nf7", {"shape": shape6, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.01, 
                        "input_noise_gain": 0.1, "sigmoid": "AbsSigmoid", "beta": 100, "theta": 1})

    # Static steps
    gi0 = GaussInput("gi0", {"shape": shape1, "sigma": (2,), "amplitude": 2, "center":(10,)})
    gi1 = GaussInput("gi1", {"shape": shape2, "sigma": (2,2,2), "amplitude": 2})

    projection1 = Projection("proj1", {"input_shape": shape2, "output_shape": shape1, "axis": (1,2), "order":(0,), "compression_type": "Sum"})
    projection0 = Projection("proj0", {"input_shape": shape1, "output_shape": (50,50), "axis": (0,), "order":(0,1), "compression_type": "Sum"})
    projection2 = Projection("proj2", {"input_shape": (50,50), "output_shape": (50,50), "axis": (0,1), "order":(1,0), "compression_type": "Sum"})

    norm0 = Normalization("norm0", {"function": "L2Norm"})
    norm1 = Normalization("norm1", {"function": "L2Norm"})

    trans0 = TransferFunction("trans0", {"function": "ExpSigmoid", "threshold": 0., "beta": 1.})

    sum0 = Sum("sum0", {})

    comp_multiply = ComponentMultiply("comp_multiply", {})

    rz1 = Resize("rz1", params={"output_shape": shape4})
    ad1 = AddConstant("ad1", params={"constant": 20})
    clmp1 = Clamp("clmp1", params={"limits": (0,1)})

    flp1 = Flip("flp1", params={"axis": (0,)})
    flp2 = Flip("flp2", params={"axis": (0,)})

    slc1 = MatrixSlice("slc1", params={"slices": ((0,10),)})
    pad1 = MatrixPadding("pad1", params={"border_size": ((0,40),),})

    vc2sc1 = VectorToScalars("vc2sc1", params={"N_scalars": 3})
    sc2vc1 = ScalarsToVector("sc2vc1", params={"N_scalars": 3})

    # Connections (different syntax possible)
    gi0 >> slc1 >> pad1 >> flp1 >> flp2 >> norm0
    gi1 >> projection1
    projection1 >> trans0
    trans0 >> norm1

    norm0 >> sum0
    norm1 >> sum0

    sum0 >> comp_multiply >> nf1

    nf1 >> projection0

    projection0 >> projection2
    projection2 >> nf2

    nf3 >> rz1 >> ad1 >> clmp1 >> nf4


    nf5 >> vc2sc1 >> nf7
    nf7 << "vc2sc1.out1"
    nf7 << "vc2sc1.out2"
    nf7 >> sc2vc1
    nf7 >> "sc2vc1.in1"
    nf7 >> "sc2vc1.in2"
    sc2vc1 >> nf6
