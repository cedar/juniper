# MatrixPadding

```python
MatrixPadding(name: str, border_size, mode: str = "constant")
```

Pads an array using `jax.numpy.pad`.

## Slots

| Slot | Description |
|------|-------------|
| Inputs | `in0` array |
| Outputs | `out0` padded array |

## Import

```python
from juniper import MatrixPadding
```

## Notes

- `border_size` follows the `pad_width` convention of `jnp.pad`.
- `mode` is passed to `jnp.pad`.
