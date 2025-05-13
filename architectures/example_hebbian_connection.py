from src.steps.GaussInput import GaussInput
from src.steps.NeuralField import NeuralField
from src.steps.HebbianConnection import HebbianConnection
from src.AbsSigmoid import AbsSigmoid
from src.GaussKernel import GaussKernel

def get_architecture(args):
    shape1 = (50,)
    shape2 = (50,)

    # Static steps
    gi0 = GaussInput("gi0", {"shape": shape1, "sigma": (2,), "amplitude": 1})
    gi1 = GaussInput("gi1", {"shape": shape2, "sigma": (2,), "amplitude": 1})
    gi2 = GaussInput("gi2", {"shape": (1,), "sigma": (0.0001,), "amplitude": 1})


    nf1 = NeuralField(f"nf1", {"shape": shape1, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.1, 
                        "input_noise_gain": 0.1, "sigmoid": AbsSigmoid(100, 0),
                        "lateral_kernel_convolution": GaussKernel({"sigma": (3,), "amplitude": 1, "normalized": True, "max_shape": shape1}),})
    nf2 = NeuralField(f"nf2", {"shape": shape2, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.1, 
                        "input_noise_gain": 0.1, "sigmoid": AbsSigmoid(100, 0),
                        "lateral_kernel_convolution": GaussKernel({"sigma": (3,), "amplitude": 1, "normalized": True, "max_shape": shape2}),})
    
    hc1 = HebbianConnection("hc1", {"shape": shape1, "target_shape": shape2, "tau": 0.1, "tau_decay": 0.1, "learning_rate": 0.1,
                        "learning_rule": "instar", "bidirectional": True, "reward_duration": (0.2,0.1)})

    # Connections (different syntax possible)
    gi0 >> nf1
    gi1 >> nf2
    gi2 >> "hc1.in2"

    nf1 >> "hc1.in0"
    nf2 >> "hc1.in1"
    
    nf2 << "hc1.out0"
    nf1 << "hc1.out1"
    