from ..configurables.Step import Step
from ..util import util
from matplotlib.colors import rgb_to_hsv
import numpy as np

def compute_kernel_factory():
    def compute_kernel(input_mats, buffer, **kwargs):
        
        # Convert to NumPy
        rgb = input_mats[util.DEFAULT_INPUT_SLOT] / 255.0  # shape (H, W, 3)
    
        # Convert to HSV
        hsv = rgb_to_hsv(np.array(rgb))  # shape (H, W, 3), values in [0,1]

        return {util.DEFAULT_OUTPUT_SLOT: hsv[:,:,0],
               "out1": hsv[:,:,1],
               "out2": hsv[:,:,2]}
    return compute_kernel


class HSV_input(Step):
    """
    Description
    ---------
    Converts an RGB image into 3 HSV channels.

    Parameters
    ---------

    Step Input/Output slots
    ---------
    - in0: jnp.array()
    - out0: jnp.array()
    - out1: jnp.array()
    - out2: jnp.array()
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = []
        super().__init__(name, params, mandatory_params)

        self.register_output("out1")
        self.register_output("out2")
        self.compute_kernel = compute_kernel_factory()