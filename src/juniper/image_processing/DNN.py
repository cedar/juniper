import logging
from ..core.backend.Exceptions import JuniperUserError

import jax
import jax.numpy as jnp
import flaxmodels as fm
import os
from ..core.frontend.Step import Step
from ..util import util
from ..util import util_jax


logger = logging.getLogger(__name__)
def compute_kernel_factory(params, model, variables):
    layer = "relu" + params["layer"]

    def vgg_maxpool_2x2(x):
        return jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 2, 2, 1),
            window_strides=(1, 2, 2, 1),
            padding="VALID",
        )
    
    if layer == "relu4_3":
        def forward(vars_, img):
                x = jnp.asarray(img, dtype=jnp.float16) / jnp.float16(255.0)
                x = jax.image.resize(x, (224, 224, 3), method="bilinear")
                x = x[None, ...]
                out = model.apply(vars_, x, train=False)
                return(out[layer][1,...])
                #return vgg_maxpool_2x2(out[layer])[1, ...] #why maxpool here
    else:
        def forward(vars_, img):
                x = jnp.asarray(img, dtype=jnp.float16) / jnp.float16(255.0)
                x = jax.image.resize(x, (224, 224, 3), method="bilinear")
                x = x[None, ...]
                out = model.apply(vars_, x, train=False)
                #return(out[layer][1,...])
                return vgg_maxpool_2x2(out[layer])[1, ...] #why maxpool here
        
    def compute_kernel(input_mats, buffer, **kwargs):
        img = input_mats[util.DEFAULT_INPUT_SLOT] 
        out = buffer[util.DEFAULT_OUTPUT_SLOT]

        img_is_empty = (img.size == 0)

        img_sum = jnp.sum(img).astype(jnp.int32)  
        key = jax.lax.cond(
            img_is_empty,
            lambda _: jnp.int32(-1),
            lambda _: img_sum,
            operand=None
        )

        out = forward(variables, img)

        """out = jax.lax.cond(
             key != buffer["lastkey"],
             lambda _: forward(variables, img), 
             lambda _: out,
             operand=None
        )"""

        return {util.DEFAULT_OUTPUT_SLOT: out, "lastkey":key}
    
    return compute_kernel

class DNN(Step):
    """
    Description
    ---------
    Applies a pretrained VGG16 model to an RGB image and returns the activation
    map of a selected convolutional layer. The input image is resized to
    (224,224,3) and normalized to [0,1] before it is passed through the model.

    Parameters
    ---------
    - layer : str("4_3", "5_3")
        - VGG16 ReLU layer suffix to read out. The full layer name is built as
          "relu" + layer.
    - model_dir (optional) : str
        - Directory used by flaxmodels to find or store VGG16 weights.
        - Default = ".flaxmodels"

    Step Input/Output slots
    ---------
    - in0: jnp.array((H,W,3))
    - out0: jnp.array(layer_shape)
        - relu4_3: (28,28,512)
        - relu5_3: (7,7,512)
    """

    _model_dir = ".flaxmodels"
    def __init__(self, name : str, layer : str, model_dir : str = _model_dir):
        params = locals().copy()
        mandatory_params = ["layer"]
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self._layer = "relu" + self._params["layer"]
        self._params["layer_shapes"] = {"relu4_3":(28,28,512), "relu5_3": (7,7,512)}
        self._params["shape"] = self._params["layer_shapes"][self._layer]

        _check_vgg16_presents(self._params, self.get_local_circuit_id())
        self._model = fm.VGG16(output="activations", include_head=False, pretrained="imagenet", ckpt_dir=self._params["model_dir"])
        self._variables = self._model.init(jax.random.PRNGKey(0), jnp.zeros((1, 224, 224, 3), dtype=jnp.float16))

        self.register_buffer("lastkey", shape=()) 
        self.buffer_map["lastkey"].dtype = jnp.int32

        self.compute_kernel = compute_kernel_factory(self._params, self._model, self._variables)

    def infer_output_shapes(self, input_specs):
         return {util.DEFAULT_OUTPUT_SLOT: self._params["shape"]}

    def infer_output_dtypes(self, input_specs):
         return {util.DEFAULT_OUTPUT_SLOT: util_jax.cfg["jdtype"]}

def _check_vgg16_presents(params, name):
    if not os.path.exists(params["model_dir"]+"/flaxmodels/vgg16_weights.h5"):
        download_request = 0
        while not download_request == 2:
            res = input(f"The DNN step '{name}' attempts to download vgg16 weights. Do you want to continue? (y/n)\n")
            if res == "y" or res=="Y":
                download_request = 2
                pass
            elif res=="n" or res=="N":
                raise JuniperUserError(f"DNN::__init__: User declined to download DNN. '{name}' unable to load vgg16 weights.")
            else:
                print("Invalid response.")
                download_request += 1
