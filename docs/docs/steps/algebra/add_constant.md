# AddConstant

Adds a scalar constant to every element of the input array.

**Type:** Static

**Import:** `from juniper import AddConstant`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `constant` | `float` | Yes | The value added to each element |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Input array |
| `out0` | Output | `(...)` | Input + constant |

## Example

```python
add5 = AddConstant("add5", {"constant": 5.0})
source >> add5
```
