from .Step import Step
from .. import util
import jax
from functools import partial
import warnings
from matplotlib.colors import rgb_to_hsv
import jax.numpy as jnp
import numpy as np
from PIL import Image

class ImageInput(Step):
    """
    Description
    ---------
    Opens an image file specified by its path.

    Parameters
    ---------
    - image_path : str

    Step Input/Output slots
    ---------
    - out0: jnp.ndarray
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ["image_path"]
        super().__init__(name, params, mandatory_params)
        
        self.is_source = True
        
        # Remove default input slot
        self.input_slot_names = []
        self._max_incoming_connections = {}

    def compute(self, input_mats, **kwargs):
        
        output = np.array(Image.open(self._params["image_path"])).astype(np.float32)

        return {util.DEFAULT_OUTPUT_SLOT: output}