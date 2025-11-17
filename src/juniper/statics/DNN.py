import jax
from functools import partial
from ..configurables.Step import Step
from ..util import util

import numpy as np
import jax.numpy as jnp
import flaxmodels as fm

class DNN(Step):

    def __init__(self, name, params):
        mandatory_params = ["layer"]
        super().__init__(name, params, mandatory_params)
        self._model = fm.VGG16(output='activations', include_head=False, pretrained='imagenet')
        self._lastsum = 0
        self._lastout = None

    #@partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        img = input_mats[util.DEFAULT_INPUT_SLOT]
        newsum = np.sum(img)
        if newsum != self._lastsum:
            img = jnp.array(img, dtype=jnp.float32) / 255.0
            img = jnp.expand_dims(img, axis=0)
            params = self._model.init(jax.random.PRNGKey(0), img)
            out = self._model.apply(params, img, train=False)
            layer = self._params["layer"]  
            self._lastsum = newsum
            self._lastout = out[layer]     
        return {util.DEFAULT_OUTPUT_SLOT: self._lastout }