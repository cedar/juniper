import jax
import jax.numpy as jnp

@jax.jit
def AbsSigmoid(x, beta, theta):
    return 0.5 * (1.0 + beta * (x - theta) / (1.0 + beta * jnp.abs(x - theta)))

@jax.jit
def ExpSigmoid(x, beta, theta):
    return 1.0 / (1.0 + jnp.exp(-beta * (x - theta)))

@jax.jit
def HeavySideSigmoid(x, beta, theta):
    return jnp.where(x < theta, 0.0, 1.0)

@jax.jit
def LinearSigmoid(x, beta, theta):
    return x * beta - theta

@jax.jit
def SemiLinearSigmoid(x, beta, theta):
    return jnp.where(x < theta, 0.0, x * beta - theta)

@jax.jit
def LogarithmicSigmoid(x, beta, theta):
    return jnp.log( beta * x - theta )
