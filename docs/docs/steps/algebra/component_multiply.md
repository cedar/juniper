# ComponentMultiply

Performs element-wise (Hadamard) multiplication of all incoming inputs. Accepts unlimited incoming connections on its default input slot. Overrides the default input summation behavior to use multiplication instead.

**Type:** Static

**Import:** `from juniper import ComponentMultiply`

## Parameters

None required.

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Accepts multiple connections (unlimited). All inputs are multiplied element-wise. |
| `out0` | Output | `(...)` | Element-wise product of all inputs |

## Example

```python
comp = ComponentMultiply("comp", {})
step_a >> comp
step_b >> comp  # output = step_a * step_b element-wise
```
