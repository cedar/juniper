import socket
import threading
import jax.numpy as jnp
import time

from ..configurables.Step import Step
from ..util import util

def cpp_crc32(data: bytes) -> int:
    # C++ style CRC32 computation
    crc = 0xFFFFFFFF
    i = 0
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
        i += 2                            # mimic the i = i + 1 bug
    return (~crc) & 0xFFFFFFFF

def serialize_cv_mat(mat: jnp.ndarray) -> bytes:
    #if not mat.flags['C_CONTIGUOUS']:
    #    mat = jnp.ascontiguousarray(mat) # jax has no flags attribute for mats
    
    if mat.dtype != jnp.float32:
        raise ValueError(f"Only CV_32F (float32) supported but received {mat.dtype}.")

    # Header construction
    dims = ",".join(str(x) for x in mat.shape)
    header = f"Mat,CV_32F,{dims},compact\n".encode("utf-8")

    # Binary data block (must be continuous!)
    binary_data = mat.tobytes()  # already correct layout

    # Combine before CRC
    message_prefix = header + binary_data
    checksum = cpp_crc32(message_prefix)

    # Final message
    footer = f"CHK-SM{checksum}E-N-D!".encode("utf-8")
    return message_prefix + footer

class TCPWriter(Step): 
    """
    Description
    ---------
    Launches a TCP write communication thread. Currently deprecated due to blocking gat/set calls.
    
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

        if 'timeout' not in params:
            self._params['timeout'] = 1.0

        if 'buffer_size' not in params:
            self._params['buffer_size'] = 32768

        if 'time_step' not in params:
            self._params['time_step'] = 1.0

        self.ip = self._params['ip']
        self.port = self._params['port']
        self.timeout = self._params['timeout']
        self.BUFFER_SIZE = self._params['buffer_size']
        self.time_step = self._params['time_step']
        self.running = True

        self.server_sock = None
        self.conn = None
        self.addr = None
        self.shape = self._params["shape"]
        self.data = jnp.zeros(self.shape)
        self.data_lock = threading.Lock()

        self.send_buffer = b''
        self.bytes_sent = 0

        self.read_buffer = b''

        self.comm_thread = threading.Thread(target=self.run, daemon=True)
        self.comm_thread.start()

    def compute(self, input_mats, **kwargs):
        self.set_data(input_mats[util.DEFAULT_INPUT_SLOT])
        return {}

    def run(self):
        while not self.establish_connection():
            continue
        
        while self.running:
            self.write()
            time.sleep(self.time_step)
    
    def establish_connection(self):
        try:
            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn.settimeout(self.timeout)
            self.conn.connect((self.ip, self.port))
            print(f"TCP write socket connected to server at ({self.ip}, {self.port})!")
            return True
        except:
            return False
    
    def write(self):
        try:
            if self.bytes_sent >= len(self.send_buffer):
                new_data = self.get_data()
                if new_data.size != 0:
                    self.send_buffer = serialize_cv_mat(new_data)
                    self.bytes_sent = 0

            if len(self.send_buffer) > 0:
                self.conn.sendall(self.send_buffer)
                self.bytes_sent += len(self.send_buffer)

        except Exception as e:
            print(e)
            print(f"TCP write socket ({self.ip},{self.port}) lost connection!")
            if self.running:
                self.close_connection()
                time.sleep(0.01)
                self.establish_connection()
    
    def close_connection(self):
        try:
            if self.conn :
                self.conn.shutdown(socket.SHUT_RDWR)
                self.conn.close()
                self.conn = None
            if self.server_sock:
                self.server_sock.close()
                self.server_sock = None
        except Exception as e:
            print(f"Error while closing the connection: {e}")

    def clean_up(self):
        self.running = False
        self.close_connection()
        self.comm_thread.join(timeout=2)
    
    def get_data (self):
        with self.data_lock:
            return self.data.copy()

    def set_data(self, data):
            with self.data_lock:
                self.data = data
