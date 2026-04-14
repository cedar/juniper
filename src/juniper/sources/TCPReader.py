import socket
import threading
import jax.numpy as jnp
import time
import re

from ..configurables.Step import Step
from ..util import util
from ..util.tcp_logging import get_tcp_logger

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
    shape = [int(dim_str) for dim_str in shape]

    dtype_map = {
        'CV_32F': jnp.float32,
        'CV_64F': jnp.float64,
        'CV_8U': jnp.uint8,
        'CV_8S': jnp.int8,
        'CV_16U': jnp.uint16,
        'CV_16S': jnp.int16,
        'CV_32S': jnp.int32,
    }

    match = re.fullmatch(r"(CV_\d+[FSU])(?:C(\d+))?", cv_type)
    if match is None:
        raise ValueError(f"Unsupported CV type: {cv_type}")

    base_cv_type = match.group(1)
    channels = int(match.group(2)) if match.group(2) is not None else 1

    if base_cv_type not in dtype_map:
        raise ValueError(f"Unsupported CV type: {cv_type}")

    dtype = dtype_map[base_cv_type]

    # Step 3: Get raw binary matrix data
    binary_data = data[header_end + 1 : trailer_start]

    # Step 4: Convert to numpy array
    mat = jnp.frombuffer(binary_data, dtype=dtype)

    # Step 5: Reshape
    expected_size = 1
    for dim in shape:
        expected_size *= dim
    expected_size *= channels

    if mat.size != expected_size:
        raise ValueError("Data size does not match header dimensions")

    if channels > 1:
        shape.append(channels)

    mat = mat.reshape(shape)
    return mat

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

        if 'timeout' not in params:
            self._params['timeout'] = 1

        if 'buffer_size' not in params:
            self._params['buffer_size'] = 32768

        if 'time_step' not in params:
            self._params['time_step'] = 1/30
        
        if 'connection_time_step' not in params:
            self._params['connection_time_step'] = 0.1
        
        
        self.is_source = True
        self.read_from_cpu = True
        self.output = jnp.zeros(self._params["shape"])
        self.register_buffer("output")

        self.compute_kernel = compute_kernel_factory()

        self.input_slot_names = []
        self._max_incoming_connections = {}

        self.ip = self._params['ip']
        self.port = self._params['port']
        self.timeout = self._params['timeout']
        self.BUFFER_SIZE = self._params['buffer_size']
        self.time_step = self._params['time_step']
        self.connection_time_step = self._params['connection_time_step']
        self.logger = get_tcp_logger(f"reader.{self._name}.{self.port}")
        self.running = True
        self._connection_announced = False

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

    def run(self):
        while self.running:
            if self.conn is None:
                while self.running and not self.establish_connection():
                    continue
                if not self.running:
                    break
            self.read()
            time.sleep(self.time_step)
    
    def establish_connection(self):
        try:
            time.sleep(self.connection_time_step)
            if self.server_sock is not None:
                self.server_sock.close()
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind((self.ip, self.port))
            self.server_sock.listen(1)
            self.server_sock.settimeout(self.timeout)
            self.conn, self.addr = self.server_sock.accept()
            self.conn.settimeout(self.timeout)
            self.logger.info(
                "Established connection on (%s,%s) with address %s",
                self.ip,
                self.port,
                self.addr,
            )
            if not self._connection_announced:
                print(f"TCP read socket ({self.ip},{self.port}) established connection with address {self.addr}!")
            self._connection_announced = True
            return True
        except Exception as e:
            self.logger.info(
                "Connection attempt on (%s,%s) failed: %r",
                self.ip,
                self.port,
                e,
            )
            return False

    def _extract_full_message(self):
        start_tag = b'Mat,'
        end_tag = b'E-N-D!'
        start = self.read_buffer.find(start_tag)
        if start == -1:
            return None

        if start > 0:
            self.read_buffer = self.read_buffer[start:]
            start = 0

        end = self.read_buffer.find(end_tag, start)
        if end == -1:
            return None

        end += len(end_tag)
        full_message = self.read_buffer[start:end]
        self.read_buffer = self.read_buffer[end:]
        return full_message
    
    def read(self):
        try:
            if self.conn is None:
                raise ConnectionError("TCP read socket is not connected")

            started_receiving = len(self.read_buffer) > 0
            deadline = time.time() + self.timeout if started_receiving else None

            while self.running:
                full_message = self._extract_full_message()
                if full_message is not None:
                    self.send_ack()
                    self.output = parse_cv_mat(full_message)
                    return

                if started_receiving and deadline is not None and time.time() >= deadline:
                    raise TimeoutError("TCP read timed out before a full message was received")

                try:
                    newdata = self.conn.recv(self.BUFFER_SIZE)
                except socket.timeout:
                    if started_receiving:
                        raise TimeoutError("TCP read timed out before a full message was received")
                    return

                if not newdata:
                    raise ConnectionError()

                started_receiving = True
                deadline = time.time() + self.timeout
                self.read_buffer += newdata

        except Exception as e:
            self.logger.exception(
                "Read loop failed on (%s,%s): %r",
                self.ip,
                self.port,
                e,
            )
            if self._connection_announced:
                print(f"TCP read socket ({self.ip},{self.port}) lost connection!")
                self._connection_announced = False
            if self.running:
                self.close_connection()
            self.read_buffer = b''
    
    def send_ack(self):
        try:
            self.conn.sendall(b'1')
        except Exception as e:
            self.logger.exception(
                "Acknowledgement failed on (%s,%s): %r",
                self.ip,
                self.port,
                e,
            )
            raise

    def close_connection(self):
        conn = self.conn
        server_sock = self.server_sock
        self.conn = None
        self.server_sock = None
        try:
            if conn:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError as e:
                    self.logger.info(
                        "Reader shutdown on (%s,%s) raised %r during close",
                        self.ip,
                        self.port,
                        e,
                    )
                conn.close()
            if server_sock:
                server_sock.close()
        except Exception as e:
            self.logger.exception(
                "Error while closing reader connection on (%s,%s): %r",
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
            return self.data.copy()

    def set_data(self, data):
            with self.data_lock:
                self.data = data
