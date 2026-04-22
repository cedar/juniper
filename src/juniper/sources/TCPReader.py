import numpy as np
from multiprocessing import Process, shared_memory

from ..configurables.Step import Step
from ..configurables.TCPWorker import TCPWorker
from ..util import util
from .. util import util_jax

def compute_kernel_factory():
    return lambda input_mats, buffer, **kwargs: {util.DEFAULT_OUTPUT_SLOT: buffer["output"], "output": buffer["output"]}

class TCPReader(Step): 
    """
    Description
    ---------
    Launches a TCP read communication thread.
    
    TODO: Remove dependency on shape and dynamic step setting. This requires reworking how the buffers and shapes re allocated and it requires rethinking how the computational graph is constructed.

    Parameters
    ----------
    - ip :
    - port : 
    - shape (optional) : (Nx,Ny,...)
    - Default = (0,)
    - timeout [s] (optional) : float
        - Time until connection times out
        - Default = 1.0
    - buffer_size [byte] (optional) : int
        - size of send packets
        - Default = 32768
    - time_step [s] (optional) : float
        - wait time between send calls
        - Default = 1.0

    Step Input/Output slots
    ----------
    - out0 : jnp.ndarray 
    """
    def __init__(self, name : str, params : dict):
        mandatory_params = ['ip', 'port', 'shape']
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.needs_input_connections = False
        self.is_source = True
        self._params["mode"] = "read"

        self.register_buffer("output")

        self.compute_kernel = compute_kernel_factory()

        shared_dtype = np.dtype(self._params.get("dtype", np.float32))
        self._params["dtype"] = shared_dtype.str
        initial_data = np.zeros(self._params["shape"], dtype=shared_dtype)
        self.shared_memory = shared_memory.SharedMemory(create=True, size=initial_data.nbytes)
        self.shared_data = np.ndarray(initial_data.shape, dtype=initial_data.dtype, buffer=self.shared_memory.buf)
        self.shared_data[:] = initial_data[:]
        self.comm_thread = Process(target=TCPWorker, args=(self._name, self._params, self.shared_memory.name))
        self.comm_thread.start()
    
    def reset(self):
        self.buffer["output"] = util_jax.ones(self._params["shape"])
        self.reset_buffer(util.DEFAULT_OUTPUT_SLOT)
        reset_state = {}
        reset_state["output"] = self.buffer["output"]  
        reset_state[util.DEFAULT_OUTPUT_SLOT] = self.buffer[util.DEFAULT_OUTPUT_SLOT]
        return reset_state

    def get_data (self):
        return_data = np.zeros(self.shared_data.shape, dtype=self.shared_data.dtype)
        return_data[:] = self.shared_data[:]
        return return_data
