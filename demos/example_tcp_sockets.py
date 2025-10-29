from juniper.steps.GaussInput import GaussInput
from juniper.steps.TCPSocket import TCPSocket
from juniper.steps.NeuralField import NeuralField



def get_architecture(args):
    # TCP Socket Steps
    tcp_reader = TCPSocket("tcp_reader", {"mode": "read", "ip": "127.0.0.1", "port": 50000, "shape": (51,51)})
    tcp_writer = TCPSocket("tcp_writer", {"mode": "write", "ip": "127.0.0.1", "port": 50001})

    # Neural Field Step
    nf1 = NeuralField(f"nf1", {"shape": (51, 51), "resting_level": -0.5, "global_inhibition": -0.0, "tau": 0.1, 
                        "input_noise_gain": 0.05, "sigmoid": "AbsSigmoid", "beta": 80, "theta":0,})
    nf2 = NeuralField(f"nf2", {"shape": (51, 51), "resting_level": -0.5, "global_inhibition": -0.0, "tau": 0.1, 
                        "input_noise_gain": 0.05, "sigmoid": "AbsSigmoid", "beta": 80, "theta":0,})
    
    # Gauss Input step
    gi1 = GaussInput("gi1", {"shape": (51, 51), "sigma": (2,2), "amplitude": 1, "center": (10,10)})

    gi1 >> nf1 >> tcp_writer
    tcp_reader >> nf2