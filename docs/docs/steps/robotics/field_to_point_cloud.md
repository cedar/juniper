# FieldToPointCloud

```python
FieldToPointCloud(name: str, origin: tuple=_origin, field_units_per_meter: tuple=_field_units_per_meter, threshold: float=_threshold, N_pt=_N_pt)
```

## Description
Converts a 3D field into a Point Cloud.

## Parameters-
- origin [m] (optional) : tuple(ox,oy,oz)
    - Origin of field with respect to origin of vector space. 
    - Default = (0,0,0)
- field_units_per_meter [1/m] (optional) : tuple(dx,dy,dz)
    - Number of field bins per meter. 
    - Default = (100,100,100)
- threshold (optional) : float
    - Only field units pircing the threshold are converted to vectors, all other are mapped to 0. 
    - Default = 0.9
- N_pt (optional) : int
    - The number of points in the cloud. 0-vectors are returned if less field points pierce the threshold.
    - Default = FieldSize

## Slots
- in0 : jnp.array((Nx,Ny,Nz))
- out0 : jnp.array((Nx*Ny*Nz,3))

## Import

```python
from juniper.robotics import FieldToPointCloud
```
