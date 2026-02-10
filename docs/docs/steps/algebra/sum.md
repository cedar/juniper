# Sum

Computes the element-wise sum of all incoming connections. Accepts unlimited incoming connections on its default input slot. The summation is performed automatically by the base `Step.update_input()` method.

**Type:** Static

**Import:** `from juniper import Sum`

## Parameters

None required.

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(...)` | Accepts multiple connections (unlimited). All inputs are summed element-wise. |
| `out0` | Output | `(...)` | Element-wise sum of all inputs |

## Example

```python
add = Sum("add", {})
step_a >> add
step_b >> add
step_c >> add  # output = step_a + step_b + step_c
```
