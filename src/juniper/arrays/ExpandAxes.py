import jax.numpy as jnp
from ..core.frontend.Step import Step
from ..util import util

def compute_kernel_factory(params):
    def compute_kernel(input_mats, buffer, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        output = jnp.expand_dims(input, axis=params["axis"])

        for ax, size in zip(params["axis"], params["sizes"]):
            output = jnp.repeat(output, size, axis=ax)
        
        #jgdb.print(f"name: {params['name']}")
        #jgdb.print(f"axis: {params['axis']}")
        #jgdb.print(f"sizes: {params['sizes']}")

        return {util.DEFAULT_OUTPUT_SLOT: output}
    return compute_kernel

class ExpandAxes(Step):
    """
    Description
    ---------
    Expand incoming step along specified axis.

    Parameters
    ---------
    - axis : tuple(ax0,ax1,...)
    - sizes : tuple(s0,s1,...)
        - sizes per dimension

    Step Input/Output slots
    ---------
    - in0 : jnp.array((Nx,...))
    - out0 : jnp.array((Nx,ax0,ax1,...))
    """
    def __init__(self, name : str, axis : tuple, sizes : tuple):
        params = locals().copy()
        mandatory_params = ["axis", "sizes"]
        super().__init__(name, params, mandatory_params)

        self.compute_kernel = compute_kernel_factory(self._params)

    def infer_output_shapes(self, input_specs):
        if util.DEFAULT_INPUT_SLOT not in input_specs:
            return {}
        shape = list(input_specs[util.DEFAULT_INPUT_SLOT][0])
        for axis, size in sorted(zip(self._params["axis"], self._params["sizes"])):
            shape.insert(axis, size)
        return {util.DEFAULT_OUTPUT_SLOT: tuple(shape)}
    
