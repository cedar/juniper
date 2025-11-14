import socket
import threading
import jax.numpy as jnp
import time

from ..configurables.Step import Step
from ..util import util

def parse_cv_mat(data: bytes):
    # Step 1: Extract header and trailer
    data_str = data.decode('latin1', errors='ignore') 
    header_end = data_str.find('\n')
    trailer_start = data_str.find('CHK-SM')

    if header_end == -1 or trailer_start == -1:
        raise ValueError("Header or trailer not found in data")

    header = data_str[:header_end]
    trailer = data_str[trailer_start:]

    # Step 2: Parse header: format is "Mat,CV_32F,rows,cols,compact"
    parts = header.strip().split(',')
    if len(parts) < 5 or parts[0] != 'Mat':
        raise ValueError("Invalid header format")

    cv_type = parts[1]
    shape = parts[2:len(parts)-1] if 'compact' in parts else parts[2:len(parts)]
    shape = jnp.array([int(dim_str) for dim_str in shape])

    dtype_map = {
        'CV_32F': jnp.float32,
        'CV_64F': jnp.float64,
        'CV_8U': jnp.uint8,
        'CV_8S': jnp.int8,
        'CV_16U': jnp.uint16,
        'CV_16S': jnp.int16,
        'CV_32S': jnp.int32,
    }

    if cv_type not in dtype_map:
        raise ValueError(f"Unsupported CV type: {cv_type}")

    dtype = dtype_map[cv_type]

    # Step 3: Get raw binary matrix data
    binary_data = data[header_end + 1 : trailer_start]

    # Step 4: Convert to numpy array
    mat = jnp.frombuffer(binary_data, dtype=dtype)

    # Step 5: Reshape
    if mat.size != jnp.prod(shape):
        raise ValueError("Data size does not match header dimensions")

    mat = mat.reshape(shape)
    return mat

class TCPReader(Step): 
    """
    Description
    ---------
    Launches a TCP read communication thread. Currently deprecated due to blocking gat/set calls.
    
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

        if 'timeout' not in params:
            self._params['timeout'] = 1

        if 'buffer_size' not in params:
            self._params['buffer_size'] = 32768

        if 'time_step' not in params:
            self._params['time_step'] = 1
        
        self.is_source = True
        self.input_slot_names = []
        self._max_incoming_connections = {}

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
        return {util.DEFAULT_OUTPUT_SLOT: self.get_data()}

    def run(self):
        while not self.establish_connection():
            continue

        while self.running:
            self.read()
            time.sleep(self.time_step)
    
    def establish_connection(self):
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.bind((self.ip, self.port))
            self.server_sock.listen(1)
            self.server_sock.settimeout(self.timeout)
            print(f"TCP read socket ({self.ip},{self.port}) waiting for connection...")
            self.conn, self.addr = self.server_sock.accept()
            print(f"TCP read socket ({self.ip},{self.port}) established connection with address {self.addr}!")
            return True
        except:
            return False
    
    def read(self):
        try:
            newdata = self.conn.recv(self.BUFFER_SIZE)
            if not newdata:
                raise ConnectionError()
            else:
                self.conn.send(b'1') 
                self.read_buffer += newdata
                start = self.read_buffer.find(b'Mat')
                end = self.read_buffer.find(b'CHK-SM')
                if start != -1 and end != -1 and end > start:
                    end_tag = b'E-N-D!'
                    end_pos = self.read_buffer.find(end_tag, end) 
                    if end_pos != -1:
                        end = end_pos + len(end_tag)
                        full_message = self.read_buffer[start:end]
                        self.read_buffer = self.read_buffer[end:]
                        self.set_data(parse_cv_mat(full_message))
        except Exception as e:
            print(e)
            print(f"TCP read socket ({self.ip},{self.port}) lost connection!")
            if self.running:
                self.close_connection()
                time.sleep(0.01)
                self.establish_connection()
            self.read_buffer = b''
    
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
