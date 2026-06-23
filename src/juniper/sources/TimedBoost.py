from ..core.frontend.Source import Source
from ..util import util
from ..util import util_jax
from ..core.backend.Exceptions import JuniperConfigurationError

def compute_kernel_factory(params, start, end, delta_t):
    def compute_kernel(input_mats, buffer, **kwargs):
        # input sum is computed in step.update_input()
        status = start < buffer["local_time"] < end

        output = params["amplitude"] * status
        # Return output
        return {util.DEFAULT_OUTPUT_SLOT: output,
                "local_time": buffer["local_time"] + delta_t}
    return compute_kernel


class TimedBoost(Source):
    """
    Description
    ---------
    Applies a homogenous boost to connected steps. Start and end of the boost can be specified.

    Parameters
    ----------
    - amplitude : float
    - duration [ms] : [start,stop]

    Step Input/Output slots
    ----------
    - out0 : jnp.ndarray((1,))
    """
    def __init__(self, name : str, amplitude : float, duration : tuple):
        params = locals().copy()
        mandatory_params = ["amplitude", "duration"]
        super().__init__(name, params, mandatory_params, is_dynamic=True)

        self._delta_t = util_jax.get_config()["delta_t"]
        
        if len(duration) != 2:
            raise JuniperConfigurationError(f"TimedBoost {name} requires a duration parameter with two values (start, end). Got: {duration} ({self.get_path_str()})")
        elif duration[0] > duration[1]:
            raise JuniperConfigurationError(f"TimedBoost {name} requires a duration parameter with start < end. Got: {duration} ({self.get_path_str()})")
        self._start = duration[0]
        self._end = duration[1]

        self.compute_kernel = compute_kernel_factory(self._params, self._start, self._end, self._delta_t)
        self.register_buffer("local_time", (1,))

    def get_data(self):
        pass
