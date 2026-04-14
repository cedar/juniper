import socket
import threading
import jax.numpy as jnp
import numpy as np
import time

from ..configurables.Step import Step
from ..util import util
from ..util.tcp_logging import get_tcp_logger

def cpp_crc32(data: bytes) -> int:
    # C++ style CRC32 computation
    crc = 0xFFFFFFFF
    """i = 0
    while i < len(data):
        b = data[i]
        # Simulate C++ signed char → int → unsigned int:
        if b > 127:
            signed = b - 256             # e.g. 194 → -62
        else:
            signed = b
        byte = signed & 0xFFFFFFFF       # two's-complement 32-bit

        crc ^= byte
        for _ in range(8):
            mask = -(crc & 1)
            crc = (crc >> 1) ^ (0xEDB88320 & mask)
        i += 1 """                           
    return (~crc) & 0xFFFFFFFF

def jnp_dtype_to_cv_type(mat: jnp.ndarray) -> str:
    base_dtype_map = {
        np.dtype(jnp.float32): "CV_32F",
        np.dtype(jnp.float64): "CV_64F",
        np.dtype(jnp.uint8): "CV_8U",
        np.dtype(jnp.int8): "CV_8S",
        np.dtype(jnp.uint16): "CV_16U",
        np.dtype(jnp.int16): "CV_16S",
        np.dtype(jnp.int32): "CV_32S",
    }

    dtype = np.dtype(mat.dtype)
    if dtype not in base_dtype_map:
        raise ValueError(f"Unsupported dtype for TCP serialization: {dtype}.")

    cv_type = base_dtype_map[dtype]
    channels = mat.shape[-1] if len(mat.shape) > 2 else 1
    if channels > 1:
        cv_type += f"C{channels}"
    return cv_type

def serialize_cv_mat(mat: jnp.ndarray) -> bytes:
    #if not mat.flags['C_CONTIGUOUS']:
    #    mat = jnp.ascontiguousarray(mat) # jax has no flags attribute for mats

    cv_type = jnp_dtype_to_cv_type(mat)

    # Header construction
    dims = mat.shape[:-1] if len(mat.shape) > 2 and cv_type.endswith(f"C{mat.shape[-1]}") else mat.shape
    header = f"Mat,{cv_type},{','.join(str(x) for x in dims)},compact\n".encode("utf-8")

    # Binary data block (must be continuous!)
    binary_data = mat.tobytes()  

    # Combine before CRC
    message_prefix = header + binary_data
    checksum = cpp_crc32(message_prefix)

    # Final message
    footer = f"CHK-SM{checksum}E-N-D!".encode("utf-8")
    return message_prefix + footer

def compute_kernel_factory(_params):
    def compute_kernel(input_mats, buffer, **kwargs): 
        input = input_mats[util.DEFAULT_INPUT_SLOT]       
        return {util.DEFAULT_OUTPUT_SLOT: input}
    return compute_kernel

class TCPWriter(Step): 
    """
    Description
    ---------
    Launches a TCP write communication thread. 
    
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
    - in0 : jnp.ndarray
    - out0 : jnp.ndarray
        - this output is mostly for debugging. May be removed in the future.
    """
    
    def __init__(self, name : str, params : dict):
        mandatory_params = ['ip', 'port']
        if "shape" not in params:
                params["shape"] = (0,)  # no output, so shape can be zero... actually, sometimes it can be necessary to have output shape (eg for debugging)
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.needs_input_connections = False
        self.is_sink = True

        if 'timeout' not in params:
            self._params['timeout'] = 10.0

        if 'buffer_size' not in params:
            self._params['buffer_size'] = 32768

        if 'time_step' not in params:
            self._params['time_step'] = 1/30

        if 'connection_time_step' not in params:
            self._params['connection_time_step'] = 1.0

        self.ip = self._params['ip']
        self.port = self._params['port']
        self.timeout = self._params['timeout']
        self.BUFFER_SIZE = self._params['buffer_size']
        self.time_step = self._params['time_step']
        self.connection_time_step = self._params['connection_time_step']
        self.logger = get_tcp_logger(f"writer.{self._name}.{self.port}")
        self.running = True
        self.connected = False
        self._connection_announced = False
        self.last_heartbeat = 0

        self.server_sock = None
        self.conn = None
        self.addr = None
        self.shape = self._params["shape"]
        self.output = jnp.zeros(self.shape)
        self.data_lock = threading.Lock()

        self.send_buffer = b''
        self.bytes_sent = 0

        self.read_buffer = b''

        self.compute_kernel = compute_kernel_factory(self._params)

        self.comm_thread = threading.Thread(target=self.run, daemon=True)
        self.comm_thread.start()

    def run(self):
        while not self.establish_connection():
            continue
        self.connected = True
        
        while self.running:
            if self.connected:
                self.write()
            else:
                self.connected = self.establish_connection()
            time.sleep(self.time_step)
    
    def establish_connection(self):
        try:
            time.sleep(self.connection_time_step)
            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn.settimeout(self.timeout)
            self.conn.connect((self.ip, self.port))
            self.logger.info(
                "Connected to server at (%s,%s)",
                self.ip,
                self.port,
            )
            if not self._connection_announced:
                print(f"TCP write socket connected to server at ({self.ip}, {self.port})!")
            self._connection_announced = True
            self.last_heartbeat = time.time()
            return True
        except Exception as e:
            self.logger.info(
                "Connection attempt to (%s,%s) failed: %r",
                self.ip,
                self.port,
                e,
            )
            return False
    
    def write(self):
        try:
            if self.bytes_sent >= len(self.send_buffer):
                self.send_buffer = serialize_cv_mat(self.output)
                self.bytes_sent = 0

            if len(self.send_buffer) > 0:
                self.conn.sendall(self.send_buffer)
                self.bytes_sent += len(self.send_buffer)
                self.send_buffer = b''

            # check alive status
            heart_beat = self.conn.recv(4096)
            if heart_beat == b"" and (time.time()-self.last_heartbeat) > self.timeout:
                raise Exception()
            else:
                self.last_heartbeat = time.time()



        except Exception as e:
            self.logger.exception(
                "Write loop failed on (%s,%s): %r",
                self.ip,
                self.port,
                e,
            )
            if self._connection_announced:
                print(f"TCP write socket ({self.ip},{self.port}) lost connection!")
                self._connection_announced = False
            if self.running:
                self.close_connection()
                time.sleep(0.01)
    
    def close_connection(self):
        try:
            if self.conn :
                self.conn.shutdown(socket.SHUT_RDWR)
                self.conn.close()
                self.conn = None
            if self.server_sock:
                self.server_sock.close()
                self.server_sock = None
            self.connected = False
        except Exception as e:
            self.connected = False
            self.logger.exception(
                "Error while closing writer connection on (%s,%s): %r",
                self.ip,
                self.port,
                e,
            )

    def clean_up(self):
        self.running = False
        self.close_connection()
        self.comm_thread.join(timeout=2)
    
    def get_data (self):
        with self.data_lock:
            return self.output.copy()

    def set_data(self, data):
            with self.data_lock:
                self.output = data
