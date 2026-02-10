# VectorsToField

Converts a set of 3D Cartesian position vectors into a 3D binary occupancy field. Each vector is mapped to a voxel index; in-bounds voxels are set to 1, out-of-bounds vectors are discarded.

**Type:** Static

**Import:** `from juniper.robotics.steps.VectorsToField import VectorsToField`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `field_shape` | `tuple(Nx, Ny, Nz)` | Yes | Shape of the output field |
| `origin` | `tuple(ox, oy, oz)` | No | World-space origin. Default: `(0, 0, 0)` |
| `field_units_per_meter` | `tuple(dx, dy, dz)` | No | Voxels per meter. Default: `(100, 100, 100)` |

## Slots

| Slot | Direction | Shape | Description |
|------|-----------|-------|-------------|
| `in0` | Input | `(N, 3)` | 3D Cartesian vectors |
| `out0` | Output | `field_shape` | Binary occupancy field |

## Example

```python
v2f = VectorsToField("v2f", {
    "field_shape": (50, 50, 50),
    "origin": (0, 0, 0),
    "field_units_per_meter": (100, 100, 100),
})
vectors >> v2f
```
