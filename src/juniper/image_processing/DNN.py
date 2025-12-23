import jax
import jax.numpy as jnp
import flaxmodels as fm
from ..configurables.Step import Step
from ..util import util

import jax.debug as jgdb

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

        out = jax.lax.cond(
             key != buffer["lastkey"],
             lambda _: forward(variables, img), 
             lambda _: out,
             operand=None
        )

        return {util.DEFAULT_OUTPUT_SLOT: out, "lastkey":key}
    
    return compute_kernel

class DNN(Step):

    def __init__(self, name, params):
        mandatory_params = ["layer"]
        params["shape"] = (0,)
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self._layer = "relu" + self._params["layer"]
        self._params["layer_shapes"] = {"relu4_3":(28,28,512), "relu5_3": (7,7,512)}
        self._params["shape"] = self._params["layer_shapes"][self._layer]
        self._model = fm.VGG16(output="activations", include_head=False, pretrained="imagenet")
        self._variables = self._model.init(jax.random.PRNGKey(0), jnp.zeros((1, 224, 224, 3), dtype=jnp.float16))

        self.register_buffer("lastkey") 

        self.compute_kernel = compute_kernel_factory(self._params, self._model, self._variables)
        self.reset()

    def reset(self):
        self.buffer["lastkey"] = jnp.int32(-1)
        self.reset_buffer(util.DEFAULT_OUTPUT_SLOT, slot_shape="shape")

        reset_state = {}
        reset_state["lastkey"] = self.buffer["lastkey"]
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self.buffer[util.DEFAULT_OUTPUT_SLOT]
        return reset_state
