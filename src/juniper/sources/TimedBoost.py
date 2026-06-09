from ..configurables.Source import Source
from ..util import util
from ..util import util_jax

def compute_kernel_factory(params, start, end, delta_t):
    def compute_kernel(input_mats, buffer, **kwargs):
        if "prng_key" not in kwargs:
            raise Exception("prng_key is a mandatory kwarg to dynamic compute()")
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
    def __init__(self, name : str, params : dict):
        mandatory_params = ["amplitude", "duration"]
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.input_slot_names = []
        self._max_incoming_connections = {}

        self._delta_t = util_jax.get_config()["delta_t"]
        
        if len(params["duration"]) != 2:
            raise ValueError(f"TimedBoost {name} requires a duration parameter with two values (start, end). Got: {params['duration']}")
        elif params["duration"][0] > params["duration"][1]:
            raise ValueError(f"TimedBoost {name} requires a duration parameter with start < end. Got: {params['duration']}")
        self._start = params["duration"][0]
        self._end = params["duration"][1]

        self.compute_kernel = compute_kernel_factory(self._params, self._start, self._end, self._delta_t)
        self.register_buffer("local_time", (1,))

    def get_data(self):
        pass
