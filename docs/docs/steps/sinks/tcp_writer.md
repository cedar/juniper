# TCPWriter

Sends matrix data to a TCP server. Launches a background thread that serializes the input matrix in CEDAR's binary format and transmits it over the network. Useful for sending data to external processes or CEDAR instances.

**Type:** Dynamic

**Import:** `from juniper.sinks.TCPWriter import TCPWriter`

**Status:** Deprecated (blocking get/set calls)

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ip` | `str` | Yes | IP address of the target server |
| `port` | `int` | Yes | Port of the target server |
| `shape` | `tuple` | No | Output buffer shape (mainly for debugging). Default: `(0,)` |
| `timeout` | `float` | No | Connection timeout in seconds. Default: `1.0` |
| `buffer_size` | `int` | No | TCP buffer size in bytes. Default: `32768` |
| `time_step` | `float` | No | Seconds between send attempts. Default: `1.0` |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Matrix data to send |
| `out0` | Output | `shape` | Debug output (may be removed in the future) |

## Example

```python
writer = TCPWriter("writer", {
    "ip": "127.0.0.1",
    "port": 5556,
})
field >> writer
```
