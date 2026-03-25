from ..configurables.Step import Step
from ..util import util
import numpy as np
from PIL import Image

def compute_kernel_factory(params):
    img = np.array(Image.open(params["image_path"])).astype(np.float32)
    def compute_kernel(input_mats, buffer, **kwargs):
        return {util.DEFAULT_OUTPUT_SLOT: img}
    return compute_kernel

class ImageLoader(Step):
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
        self.compute_kernel = compute_kernel_factory(self._params)