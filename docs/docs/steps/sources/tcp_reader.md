# TCPReader

Receives matrix data from a TCP connection. Launches a background thread that listens for incoming data in CEDAR's binary matrix format. Useful for receiving data from external processes or CEDAR instances.

**Type:** Dynamic (Source)

**Import:** `from juniper.sources.TCPReader import TCPReader`

**Status:** Deprecated (blocking get/set calls)

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ip` | `str` | Yes | IP address to bind to |
| `port` | `int` | Yes | Port to listen on |
| `shape` | `tuple` | Yes | Expected shape of incoming data |
| `timeout` | `float` | No | Connection timeout in seconds. Default: `1.0` |
| `buffer_size` | `int` | No | TCP buffer size in bytes. Default: `32768` |
| `time_step` | `float` | No | Seconds between read attempts. Default: `1.0` |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `out0` | Output | `shape` | Last received matrix |

## Example

```python
reader = TCPReader("reader", {
    "ip": "127.0.0.1",
    "port": 5555,
    "shape": (50, 50),
})
reader >> field
```
