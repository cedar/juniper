# PointCloudToField

```python
PointCloudToField(name: str, field_shape: tuple, origin: tuple=_origin, field_units_per_meter: tuple=_field_units_per_meter)
```

## Description
Converts a 3D PointCloud into a 3D field like representation (Nx, Ny, Nz).

## Parameters-
- field_shape : tuple(Nx,Ny,Nz)
- origin [m] (optional) : tuple(ox,oy,oz)
    - Origin of field with respect to origin of vector space. 
    - Default = (0,0,0)
- field_units_per_meter [1/m] (optional) : tuple(dx,dy,dz)
    - Number of field bins per meter. 
    - Default = (100,100,100)

## Slots-
- in0 : jnp.ndarray((N,3))
- out0 : jnp.ndarray(field_shape)

## Import

```python
from juniper.robotics import PointCloudToField
```
