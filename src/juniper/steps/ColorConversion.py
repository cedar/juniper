from .Step import Step
from .. import util
import jax
from functools import partial
import warnings
from matplotlib.colors import rgb_to_hsv
import jax.numpy as jnp
import numpy as np

class ColorConversion(Step):

    def __init__(self, name, params):
        mandatory_params = []
        super().__init__(name, params, mandatory_params)

        self.register_output("out1")
        self.register_output("out2")

    def compute(self, input_mats, **kwargs):
        
        # Convert to NumPy
        rgb = input_mats[util.DEFAULT_INPUT_SLOT] / 255.0  # shape (H, W, 3)
    
        # Convert to HSV
        hsv = rgb_to_hsv(np.array(rgb))  # shape (H, W, 3), values in [0,1]

        return {util.DEFAULT_OUTPUT_SLOT: hsv[:,:,0],
               "out1": hsv[:,:,1],
               "out2": hsv[:,:,2]}