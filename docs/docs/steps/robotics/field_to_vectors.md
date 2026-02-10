# FieldToVectors

Converts a 3D neural field into a set of 3D Cartesian position vectors. Only voxels above the threshold are converted; all others are mapped to zero. The voxel-to-world mapping is controlled by `origin` and `field_units_per_meter`.

**Type:** Static

**Import:** `from juniper.robotics.steps.FieldToVectors import FieldToVectors`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `origin` | `tuple(ox, oy, oz)` | No | World-space origin of the field. Default: `(0, 0, 0)` |
| `field_units_per_meter` | `tuple(dx, dy, dz)` | No | Voxels per meter in each dimension. Default: `(100, 100, 100)` |
| `threshold` | `float` | No | Only voxels above this value produce vectors. Default: `0.9` |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(Nx, Ny, Nz)` | 3D field |
| `out0` | Output | `(Nx*Ny*Nz, 3)` | Cartesian position vectors |

## Example

```python
f2v = FieldToVectors("f2v", {
    "origin": (0, 0, 0),
    "field_units_per_meter": (100, 100, 100),
    "threshold": 0.9,
})
field >> f2v
```
