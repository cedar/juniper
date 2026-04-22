from .Configurable import Configurable
from ..util.util_tcp import get_tcp_logger

import numpy as np
import re, time, socket
from multiprocessing import shared_memory

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

def serialize_cv_mat(mat: np.ndarray) -> bytes:
    #if not mat.flags['C_CONTIGUOUS']:
    #    mat = jnp.ascontiguousarray(mat) # jax has no flags attribute for mats
    base_dtype_map = {
        np.dtype(np.float32): "CV_32F",
        np.dtype(np.float64): "CV_64F",
        np.dtype(np.uint8): "CV_8U",
        np.dtype(np.int8): "CV_8S",
        np.dtype(np.uint16): "CV_16U",
        np.dtype(np.int16): "CV_16S",
        np.dtype(np.int32): "CV_32S",
    }

    dtype = np.dtype(mat.dtype)
    if dtype not in base_dtype_map:
        raise ValueError(f"Unsupported dtype for TCP serialization: {dtype}.")

    cv_type = base_dtype_map[dtype]
    if mat.ndim == 3 and mat.shape[-1] == 3:
        cv_type += "C3"

    # Header construction
    dims = mat.shape[:-1] if (mat.ndim == 3 and mat.shape[-1] == 3) else mat.shape
    header = f"Mat,{cv_type},{','.join(str(x) for x in dims)},compact\n".encode("utf-8")

    # Binary data block 
    binary_data = mat.tobytes()  

    # Combine before CRC
    message_prefix = header + binary_data
    checksum = cpp_crc32(message_prefix)

    # Final message
    footer = f"CHK-SM{checksum}E-N-D!".encode("utf-8")
    return message_prefix + footer

def _parse_cv_header(header_bytes: bytes):
    header = header_bytes.decode("utf-8", errors="strict").strip()
    parts = header.split(",")
    if len(parts) < 5 or parts[0] != "Mat":
        raise ValueError("Invalid header format")

    cv_type = parts[1]
    shape = [int(dim_str) for dim_str in parts[2:-1]]

    dtype_map = {
        "CV_32F": np.float32,
        "CV_64F": np.float64,
        "CV_8U": np.uint8,
        "CV_8S": np.int8,
        "CV_16U": np.uint16,
        "CV_16S": np.int16,
        "CV_32S": np.int32,
    }

    match = re.fullmatch(r"(CV_\d+[FSU])(?:C(\d+))?", cv_type)
    if match is None:
        raise ValueError(f"Unsupported CV type: {cv_type}")

    base_cv_type = match.group(1)
    channels = int(match.group(2)) if match.group(2) is not None else 1

    if base_cv_type not in dtype_map:
        raise ValueError(f"Unsupported CV type: {cv_type}")

    dtype = np.dtype(dtype_map[base_cv_type])
    expected_size = channels
    for dim in shape:
        expected_size *= dim

    return shape, dtype, channels, expected_size * dtype.itemsize

def parse_cv_mat(data: bytes):
    header_end = data.find(b"\n")
    trailer_start = data.find(b"CHK-SM", header_end + 1)

    if header_end == -1 or trailer_start == -1:
        raise ValueError("Header or trailer not found in data")

    shape, dtype, channels, payload_nbytes = _parse_cv_header(data[:header_end])
    binary_data = memoryview(data)[header_end + 1 : trailer_start]

    if len(binary_data) != payload_nbytes:
        raise ValueError("Data size does not match header dimensions")

    mat = np.frombuffer(binary_data, dtype=dtype)

    if channels > 1:
        shape.append(channels)

    return mat.reshape(shape)

class TCPWorker(Configurable):
    def __init__(self, name : str, params : dict, shared_memory_name):
        mandatory_params = ['ip', 'port', 'shape', 'mode']
        self._name = name
        super().__init__(params, mandatory_params)
        #self._params = params.copy()

        if 'timeout' not in params:
            self._params['timeout'] = 3

        if 'buffer_size' not in params:
            self._params['buffer_size'] = 32768

        if 'time_step' not in params:
            self._params['time_step'] = 1/30

        if 'connect_retry_delay' not in params:
            self._params['connect_retry_delay'] = max(self._params['time_step'], 0.5)

        if 'max_missed_heartbeats' not in params:
            self._params['max_missed_heartbeats'] = 3

        if 'send_on_change_only' not in params:
            self._params['send_on_change_only'] = True
        
        self.shared_memory = shared_memory.SharedMemory(name=shared_memory_name)
        self.shared_dtype = np.dtype(self._params.get("dtype", np.float32))
        self.shared_data = np.ndarray(self._params["shape"], dtype=self.shared_dtype, buffer=self.shared_memory.buf)

        self.ip = self._params['ip']
        self.port = self._params['port']
        self.timeout = self._params['timeout']
        self.BUFFER_SIZE = self._params['buffer_size']
        self.logger = get_tcp_logger(f"writer.{self._name}.{self.port}") if self._params["mode"] == "write" else get_tcp_logger(f"reader.{self._name}.{self.port}")
        self._connection_announced = False
        self.time_step = self._params['time_step']
        self.connect_retry_delay = self._params['connect_retry_delay']
        self.max_missed_heartbeats = self._params['max_missed_heartbeats']
        self.send_on_change_only = self._params['send_on_change_only']
        self.last_heartbeat = 0
        self.missed_heartbeats = 0

        self.server_sock = None
        self.conn = None
        self.addr = None
        self.shape = self._params["shape"]

        self.send_buffer = b''
        self.read_buffer = bytearray()
        self.bytes_sent = 0
        self._last_sent_snapshot = None
        self._logged_send_header = False


        self.running = True
        self.run(self.write if self._params["mode"] == "write" else self.read, 
                 self.establish_write_connection if self._params["mode"] == "write" else self.establish_read_connection)

        
    def run(self, comm_protocol, conn_protocol):
        while self.running:
            if self.conn is None:
                while self.running and not conn_protocol():
                    continue
                if not self.running:
                    break
            comm_protocol()
            time.sleep(self.time_step)
    
    def establish_write_connection(self):
        try:
            time.sleep(self.connect_retry_delay)
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
            self.missed_heartbeats = 0
            self.send_buffer = b''
            self.bytes_sent = 0
            self._logged_send_header = False
            return True
        except Exception as e:
            self.logger.info(
                "Connection attempt to (%s,%s) failed: %r",
                self.ip,
                self.port,
                e,
            )
            return False

    def establish_read_connection(self):
        try:
            time.sleep(self.connect_retry_delay)
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
            self.missed_heartbeats = 0
            self.read_buffer.clear()
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
        start_tag = b"Mat,"
        footer_tag = b"CHK-SM"
        end_tag = b"E-N-D!"

        start = self.read_buffer.find(start_tag)
        if start == -1:
            if len(self.read_buffer) > len(start_tag):
                del self.read_buffer[:-len(start_tag)]
            return None

        if start > 0:
            del self.read_buffer[:start]

        header_end = self.read_buffer.find(b"\n")
        if header_end == -1:
            return None

        _, _, _, payload_nbytes = _parse_cv_header(bytes(self.read_buffer[:header_end]))
        payload_start = header_end + 1
        footer_start = payload_start + payload_nbytes

        if len(self.read_buffer) < footer_start + len(footer_tag) + len(end_tag):
            return None

        if bytes(self.read_buffer[footer_start:footer_start + len(footer_tag)]) != footer_tag:
            raise ValueError("Footer not found after payload")

        end = self.read_buffer.find(end_tag, footer_start + len(footer_tag))
        if end == -1:
            return None

        end += len(end_tag)
        full_message = bytes(self.read_buffer[:end])
        del self.read_buffer[:end]
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
                    data = parse_cv_mat(full_message)
                    self.shared_data[:] = data[:]
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
                self.read_buffer.extend(newdata)

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
            self.read_buffer.clear()
    
    def write(self):
        try:
            data = np.empty(self._params["shape"], dtype=self.shared_dtype)
            data[:] = self.shared_data[:]
            data_bytes = data.tobytes()

            if self.send_on_change_only and self._last_sent_snapshot == data_bytes:
                return

            self.send_buffer = serialize_cv_mat(data)
            self.bytes_sent = 0

            if not self._logged_send_header:
                header_end = self.send_buffer.find(b"\n")
                if header_end != -1:
                    self.logger.info(
                        "Sending header to (%s,%s): %s",
                        self.ip,
                        self.port,
                        self.send_buffer[:header_end].decode("utf-8", errors="replace"),
                    )
                    self._logged_send_header = True

            if len(self.send_buffer) > 0:
                self.conn.sendall(self.send_buffer)
                self.bytes_sent += len(self.send_buffer)
                self.send_buffer = b''
                self._last_sent_snapshot = data_bytes

            # check alive status
            try:
                heart_beat = self.conn.recv(4096)
            except socket.timeout:
                self.missed_heartbeats += 1
                if self.missed_heartbeats < self.max_missed_heartbeats:
                    return
                raise
            if heart_beat == b"" and (time.time()-self.last_heartbeat) > self.timeout:
                raise Exception()
            else:
                self.last_heartbeat = time.time()
                self.missed_heartbeats = 0
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
                self._last_sent_snapshot = None
                self.missed_heartbeats = 0
                self.send_buffer = b''
                self.bytes_sent = 0
                self.close_connection()
                time.sleep(self.connect_retry_delay)

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
        self.read_buffer.clear()
        self.send_buffer = b''
        self.bytes_sent = 0
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
   
