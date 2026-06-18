import numpy as np
from multiprocessing import Process, shared_memory

from ..core.frontend.Sink import Sink
from ..util.TCPWorker import TCPWorker
from ..util import util

def compute_kernel_factory(_params):
    def compute_kernel(input_mats, buffer, **kwargs): 
        input = input_mats[util.DEFAULT_INPUT_SLOT]       
        return {util.DEFAULT_OUTPUT_SLOT: input}
    return compute_kernel

class TCPWriter(Sink): 
    """
    Description
    ---------
    Launches a TCP write communication thread.

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
    - in0 : jnp.ndarray
    - out0 : jnp.ndarray
        - this output is mostly for debugging. May be removed in the future.
    """
    
    def __init__(self, name : str, params : dict):
        mandatory_params = ['ip', 'port', 'shape']
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.needs_input_connections = False
        self._params["mode"] = "write"

        self.compute_kernel = compute_kernel_factory(self._params)

        shared_dtype = np.dtype(self._params.get("dtype", np.float32))
        self._params["dtype"] = shared_dtype.str
        initial_data = np.zeros(self._params["shape"], dtype=shared_dtype)
        self.shared_memory = shared_memory.SharedMemory(create=True, size=initial_data.nbytes)
        self.shared_data = np.ndarray(initial_data.shape, dtype=initial_data.dtype, buffer=self.shared_memory.buf)
        self.shared_data[:] = initial_data[:]
        self.comm_thread = Process(target=TCPWorker, args=(self.get_local_circuit_id(), self._params, self.shared_memory.name))

    def close(self):
        if self.comm_thread.is_alive():
            self.comm_thread.kill()
            self.comm_thread.join(timeout=1.)
        self.comm_thread.close()
        self.shared_memory.close()
        self.shared_memory.unlink()

    def open(self):
        if not self.comm_thread.is_alive():
            self.comm_thread.start()

    def set_data(self, data):
        data = np.asanyarray(data, dtype=self.shared_data.dtype)
        if data.shape != self.shared_data.shape:
            if data.size != self.shared_data.size:
                raise ValueError(
                    f"TCPWriter {self._name} received shape {data.shape}, expected {self.shared_data.shape}."
                )
            data = data.reshape(self.shared_data.shape)
        self.shared_data[:] = data[:]
