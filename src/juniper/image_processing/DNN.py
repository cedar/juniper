import jax
import jax.numpy as jnp
import flaxmodels as fm
from ..configurables.Step import Step
from ..util import util


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
    
    def forward(vars_, x):
            out = model.apply(vars_, x, train=False)
            return vgg_maxpool_2x2(out[layer])[1, ...]
    
    def compute_kernel(input_mats, buffer, **kwargs):
        img = input_mats[util.DEFAULT_INPUT_SLOT] 
        key = (img.shape, int(img.sum()))
        if key != buffer["lastkey"]:
            x = jnp.asarray(img, dtype=jnp.float16) / jnp.float16(255.0)
            x = jax.image.resize(x, (224, 224, 3), method="bilinear")
            x = x[None, ...]
            out = forward(variables, x)
        return {util.DEFAULT_OUTPUT_SLOT: out, "lastkey":key}
    
    return compute_kernel

class DNN(Step):

    def __init__(self, name, params):
        mandatory_params = ["layer"]
        super().__init__(name, params, mandatory_params)
        self._layer = "relu" + self._params["layer"]
        self._model = fm.VGG16(output="activations", include_head=False, pretrained="imagenet")
        self._variables = self._model.init(jax.random.PRNGKey(0), jnp.zeros((1, 224, 224, 3), dtype=jnp.float16))

        self.register_buffer("lastkey") 

        self.compute_kernel = compute_kernel_factory(self._params, self._model, self._variables)

    def reset(self):
        self.buffer["lastkey"] = jnp.zeros((1,))
        self.reset_buffer(util.DEFAULT_OUTPUT_SLOT)

        reset_state = {}
        reset_state["lastkey"] = self.buffer["lastkey"]
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self.buffer[util.DEFAULT_OUTPUT_SLOT]
        return reset_state
