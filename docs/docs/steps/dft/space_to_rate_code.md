# SpaceToRateCode

Extracts the peak position from a field-like array and outputs it as a rate-coded vector. Assumes at most one peak in the input field. Values above the threshold are averaged to determine the peak location.

**Type:** Dynamic

**Import:** `from juniper import SpaceToRateCode`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `shape` | `tuple` | Yes | Shape of the input field |
| `limits` | `tuple((low, high), ...)` | Yes | Metric range per dimension |
| `tau` | `float` | No | If set, output converges exponentially to the peak. If `0`, jumps instantly. Default: `0` |
| `cyclic` | `bool` | No | Cyclic mode (not yet implemented). Default: `False` |
| `threshold` | `float` | No | Minimum activation to count as a peak. Default: `0.9` |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `shape` | Input field |
| `out0` | Output | `(len(shape),)` | Peak position as a vector |

## Example

```python
s2r = SpaceToRateCode("s2r", {
    "shape": (50, 50),
    "limits": ((0, 50), (0, 50)),
    "threshold": 0.9,
})
neural_field >> s2r
```
