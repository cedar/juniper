import jax.numpy as jnp
from jax import lax
import jax.scipy as jsp

def correlate1d_jax(x, w, axis=-1, mode="constant", cval=0.0, origin=0):
    """
    Minimal correlate1d in JAX:
    - SAME-shape output
    - along 'axis'
    - supports modes via jnp.pad
    - 'origin' alignment like SciPy
    Assumes real inputs and valid args.
    """
    x = jnp.asarray(x)
    w = jnp.asarray(w)
    L = int(w.shape[0])

    # pad split that encodes origin (no validation here)
    pad_left = (L - 1) // 2 + int(origin)
    pad_right = (L - 1) - pad_left

    # move target axis to the end, pad only there
    if axis < 0: 
        axis += x.ndim
    xT = jnp.moveaxis(x, axis, -1)
    pad_width = [(0, 0)] * (x.ndim - 1) + [(pad_left, pad_right)]

    # only pass constant_values for constant mode (JAX requirement)
    pad_kwargs = {}
    if mode == "constant":
        pad_kwargs["constant_values"] = cval

    x_pad = jnp.pad(xT, pad_width, mode=mode, **pad_kwargs)

    # 1D correlation via lax.conv (which is correlation, not convolution)
    xN = x_pad.reshape((-1, x_pad.shape[-1], 1))   # [N, L, C=1]
    k = w.reshape((-1, 1, 1))                      # [W, I=1, O=1]
    y = lax.conv_general_dilated(
        lhs=xN, rhs=k,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    y = y.reshape(x_pad.shape[:-1] + (y.shape[1],))
    return jnp.moveaxis(y, -1, axis)

def convolve1d_jax(x, w, axis=-1, mode="constant", cval=0.0, origin=0):
    """
    Minimal convolve1d in JAX:
    implemented via correlate1d_jax with flipped kernel + origin tweak
    (same trick SciPy uses).
    """
    w = jnp.asarray(w)[::-1]
    new_origin = -int(origin)
    if (w.shape[0] % 2) == 0:
        new_origin -= 1
    return correlate1d_jax(x, w, axis=axis, mode=mode, cval=cval, origin=new_origin)

def factorized_convolve(x, k, mode="constant", cval=0.0, origin=0):
    for axis, wk in enumerate(k):
        x = convolve1d_jax(x, wk, axis=axis, mode=mode, cval=cval, origin=origin)
    return x

def full_convolve(x, k, mode="same", cval=None, origin=None):
    return jsp.signal.fftconvolve(x, k, mode=mode)       

def convolve_func_singleton(kernel, factorized):
    if not factorized:
        # full kernel: if ndarray, wrap as single component; if list of ndarrays, keep
        if isinstance(kernel, jnp.ndarray):
            kernel_obj = (kernel,)
        else:
            # if someone passed a Python list of components
            kernel_obj = tuple(kernel)
    else:
        # factorized kernel
        # Detect whether `kernel` is a single component (list of 1D arrays)
        # vs multiple components (list of list-of-1D arrays).
        if len(kernel) > 0 and isinstance(kernel[0], jnp.ndarray) and kernel[0].ndim == 1:
            # single component: [k0, k1, ...]
            kernel_obj = (tuple(kernel),)
        else:
            # multiple components: [[k0,k1,...], [k0,k1,...], ...]
            kernel_obj = tuple(tuple(comp) for comp in kernel)
        

    conv_funcs = []
    for kernel_component in kernel_obj:
        if factorized:
            component_modes = tuple(kernel_component)
            conv_funcs.append(lambda x, k=component_modes: factorized_convolve(x, k))
        else:
            component_full = jnp.asarray(kernel_component)
            conv_funcs.append(lambda x, k=component_full: full_convolve(x, k))
    conv_funcs = tuple(conv_funcs)
        
    def convolve(x):
        out = jnp.zeros_like(x)
        for conv_func in conv_funcs:
            out += conv_func(x)
        return out
    
    return convolve