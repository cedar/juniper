# TCPReader

```python
TCPReader(name: str, ip: str, port: int, shape: tuple, dtype=_dtype, timeout: float=_timeout, buffer_size: int=_buffer_size, time_step: float=_time_step, connect_retry_delay=_connect_retry_delay, max_missed_heartbeats: int=_max_missed_heartbeats, send_on_change_only: bool=_send_on_change_only)
```

## Description
Launches a TCP read communication thread.

## Parameters-
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

## Slots-
- out0 : jnp.ndarray

## Import

```python
from juniper import TCPReader
```

## Example

```python
reader = TCPReader("camera", ip="127.0.0.1", port=5000, shape=(480, 640, 3))
```
