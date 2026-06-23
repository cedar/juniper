import logging
import numpy as np
from multiprocessing import Process, shared_memory, Queue
from logging.handlers import QueueListener
import logging

from ..core.frontend.Source import Source
from ..util.TCPWorker import TCPWorker
from ..util import util


logger = logging.getLogger(__name__)
class ForwardingHandler(logging.Handler):
    def __init__(self, logger_name: str):
        super().__init__()
        self.target_logger = logging.getLogger(logger_name)

    def emit(self, record):
        self.target_logger.handle(record)

def compute_kernel_factory():
    return lambda input_mats, buffer, **kwargs: {util.DEFAULT_OUTPUT_SLOT: buffer[util.DEFAULT_OUTPUT_SLOT]}

class TCPReader(Source): 
    """
    Description
    ---------
    Launches a TCP read communication thread.
    
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
    _dtype = np.float32
    _timeout = 3
    _buffer_size = 32768
    _time_step = 1/30
    _connect_retry_delay = None
    _max_missed_heartbeats = 3
    _send_on_change_only = True
    def __init__(
            self,
            name : str,
            ip : str,
            port : int,
            shape : tuple,
            dtype = _dtype,
            timeout : float = _timeout,
            buffer_size : int = _buffer_size,
            time_step : float = _time_step,
            connect_retry_delay = _connect_retry_delay,
            max_missed_heartbeats : int = _max_missed_heartbeats,
            send_on_change_only : bool = _send_on_change_only):
        params = locals().copy()
        mandatory_params = ['ip', 'port', 'shape']
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.needs_input_connections = False
        self._params["mode"] = "read"
        if connect_retry_delay is None:
            self._params["connect_retry_delay"] = max(time_step, 0.5)

        self.compute_kernel = compute_kernel_factory()

        # logging
        queue = Queue(maxsize=42)
        handler = ForwardingHandler("juniper")
        self.logger = QueueListener(queue, handler, respect_handler_level=True)
        self.logger.start()

        # TCP loop process
        shared_dtype = np.dtype(self._params.get("dtype", np.float32))
        self._params["dtype"] = shared_dtype.str
        initial_data = np.zeros(self._params["shape"], dtype=shared_dtype)
        self.shared_memory = shared_memory.SharedMemory(create=True, size=initial_data.nbytes)
        self.shared_data = np.ndarray(initial_data.shape, dtype=initial_data.dtype, buffer=self.shared_memory.buf)
        self.shared_data[:] = initial_data[:]
        self.comm_thread = Process(target=TCPWorker, args=(self.get_path_str(), self._params, self.shared_memory.name, queue))
        #self.comm_thread.start()

    def close(self):
        if self.comm_thread.is_alive():
            self.comm_thread.kill()
            self.comm_thread.join(timeout=1.)
        self.shared_memory.close()
        self.shared_memory.unlink()
        self.logger.stop()
    
    def open(self):
        if not self.comm_thread.is_alive():
            self.comm_thread.start()

    def get_data(self):
        return_data = np.zeros(self.shared_data.shape, dtype=self.shared_data.dtype)
        return_data[:] = self.shared_data[:]
        return return_data

    def infer_output_shapes(self, input_specs):
        return {util.DEFAULT_OUTPUT_SLOT: tuple(self._params["shape"])}

    def infer_output_dtypes(self, input_specs):
        return {util.DEFAULT_OUTPUT_SLOT: np.dtype(self._params["dtype"]).type}
