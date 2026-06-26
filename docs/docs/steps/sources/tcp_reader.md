# TCPReader

```python
TCPReader(
    name: str,
    ip: str,
    port: int,
    shape: tuple,
    dtype=numpy.float32,
    timeout: float = 3,
    buffer_size: int = 32768,
    time_step: float = 1 / 30,
    connect_retry_delay=None,
    max_missed_heartbeats: int = 3,
    send_on_change_only: bool = True,
)
```

Receives arrays from a TCP connection through a worker process.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | No input slots |
| Outputs | `out0` array with configured `shape` and `dtype` |

## Import

```python
from juniper import TCPReader
```

## Notes

- Call `arch.close_connections()` after TCP runs in long-lived Python processes.
- If `connect_retry_delay` is omitted, JUNIPER uses at least the configured `time_step`.
