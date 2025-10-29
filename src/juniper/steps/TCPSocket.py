import socket
import threading
import jax.numpy as jnp
import jax.debug as jdbg

import numpy as np
import time

from .Step import Step
from .. import util

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

class TCPSocket(Step): # TODO: Remove dependency on shape and dynamic step setting. This requires reworking how the buffers and shapes re allocated and it requires rethinking how the computational graph is constructed.
    # This class adds a TCP socket communication step to Juniper. It can operate in either 'read' or 'write' mode. Sockets are handled in separate threads to avoid blocking the main computation.
    # In 'read' mode, it listens for incoming TCP connections and reads data from the socket, making it available as output.
    # In 'write' mode, it connects to a specified TCP server and sends data received from its input.
    # Currently this step is primarily for communication with Cedar. The checksum computation and format are specific to work with Cedar's TCP read and write sockets.
    def __init__(self, name, params):
        mandatory_params = ['ip', 'port', 'mode']
        if params["mode"] == "write":
                params["shape"] = (0,)  # in write mode, no output, so shape can be zero
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.needs_input_connections = False

        if 'timeout' not in params:
            params['timeout'] = 1

        if 'buffer_size' not in params:
            params['buffer_size'] = 32768

        if 'time_step' not in params:
            params['time_step'] = 0.02

        if params['mode'] not in ['read', 'write']:
            raise ValueError(f"TCPSocket {name} has invalid mode {params['mode']}. Must be 'read' or 'write'.")
        
        if params['mode'] == 'read':
            self.is_source = True
            self.input_slot_names = []
            self._max_incoming_connections = {}

        self.ip = params['ip']
        self.port = params['port']
        self.timeout = params['timeout']
        self.BUFFER_SIZE = params['buffer_size']
        self.time_step = params['time_step']
        self.mode = params['mode']
        self.running = True

        self.server_sock = None
        self.conn = None
        self.addr = None
        self.data = jnp.zeros((10,10))
        self.data_lock = threading.Lock()

        self.send_buffer = b''
        self.bytes_sent = 0

        self.read_buffer = b''

        self.comm_thread = threading.Thread(target=self.run, daemon=True)
        self.comm_thread.start()

    def compute(self, input_mats, **kwargs):
        if self.mode == 'write':
            self.set_data(input_mats[util.DEFAULT_INPUT_SLOT])
        return {util.DEFAULT_OUTPUT_SLOT: self.get_data()} if self.mode == 'read' else {}

    def run(self):
        while not self.establish_connection():
            continue

        if self.mode == "read":
            while self.running:
                self.read()
                time.sleep(self.time_step)
        elif self.mode == "write":
            while self.running:
                self.write()
                time.sleep(self.time_step)
    
    def establish_connection(self):
        try:
            if self.mode == "read":
                self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_sock.bind((self.ip, self.port))
                self.server_sock.listen(1)
                self.server_sock.settimeout(self.timeout)
                print(f"TCP {self.mode} socket ({self.ip},{self.port}) waiting for connection...")
                self.conn, self.addr = self.server_sock.accept()
                print(f"TCP {self.mode} socket ({self.ip},{self.port}) established connection with address {self.addr}!")
            elif self.mode == "write":
                self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.conn.settimeout(self.timeout)
                self.conn.connect((self.ip, self.port))
                print(f"TCP {self.mode} socket connected to server at ({self.ip}, {self.port})!")
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
            print(f"TCP {self.mode} socket ({self.ip},{self.port}) lost connection!")
            if self.running:
                self.close_connection()
                time.sleep(0.01)
                self.establish_connection()
            self.read_buffer = b''
    
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
            print(f"TCP {self.mode} socket ({self.ip},{self.port}) lost connection!")
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
