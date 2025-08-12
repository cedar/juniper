from src.steps.Step import Step
from src import util
import jax
from functools import partial
from src import util_jax

class TimedBoost(Step):

    def __init__(self, name, params):
        mandatory_params = ["amplitude", "duration"]
        params["shape"] = (1,)
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.is_source = True
        self.input_slot_names = []
        self._max_incoming_connections = {}

        self._delta_t = util_jax.get_config()["delta_t"]
        
        if len(params["duration"]) != 2:
            raise ValueError(f"TimedBoost {name} requires a duration parameter with two values (start, end). Got: {params['duration']}")
        elif params["duration"][0] > params["duration"][1]:
            raise ValueError(f"TimedBoost {name} requires a duration parameter with start < end. Got: {params['duration']}")
        self._start = params["duration"][0]
        self._end = params["duration"][1]

        self.reset()

    # required kwargs are: delta_t, prng_key
    def compute(self, input_mats, **kwargs):
        if not "prng_key" in kwargs:
            raise Exception("prng_key is a mandatory kwarg to dynamic compute()")
        # input sum is computed in step.update_input()
        status = self._start < self.buffer["local_time"] < self._end

        output = self._params["amplitude"] * status
        # Return output
        return {util.DEFAULT_OUTPUT_SLOT: output,
                "local_time": self.buffer["local_time"] + self._delta_t}
    
    def reset(self): # Override
        self.buffer["local_time"] = util_jax.ones(self._params["shape"])*0
        self.reset_buffer(util.DEFAULT_OUTPUT_SLOT)
