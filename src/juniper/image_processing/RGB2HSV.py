import logging
from ..core.frontend.Step import Step
from ..util import util
from .ColorConversion import rgb_to_hsv_jax


logger = logging.getLogger(__name__)
def compute_kernel_factory():
    def compute_kernel(input_mats, buffer, **kwargs):
        rgb = input_mats[util.DEFAULT_INPUT_SLOT] / 255.0  # shape (H, W, 3)
    
        # Convert to HSV
        hsv = rgb_to_hsv_jax(rgb, channels="rgb")  # shape (H, W, 3), values in [0,1]

        return {util.DEFAULT_OUTPUT_SLOT: hsv[:,:,0],
               "out1": hsv[:,:,1],
               "out2": hsv[:,:,2]}
    return compute_kernel


class RGB2HSV(Step):
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
    def __init__(self, name : str):
        params = locals().copy()
        mandatory_params = []
        super().__init__(name, params, mandatory_params)

        self.register_output_slot("out1")
        self.register_output_slot("out2")
        self.compute_kernel = compute_kernel_factory()

    def infer_output_shapes(self, input_specs):
        if util.DEFAULT_INPUT_SLOT not in input_specs:
            return {}
        input_shape = tuple(input_specs[util.DEFAULT_INPUT_SLOT][0])
        if len(input_shape) == 0:
            return {}
        output_shape = input_shape[:-1]
        return {
            util.DEFAULT_OUTPUT_SLOT: output_shape,
            "out1": output_shape,
            "out2": output_shape,
        }
