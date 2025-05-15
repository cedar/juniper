import jax
import jax.numpy as jnp

@jax.jit
def AbsSigmoid(x, beta, theta):
    return 0.5 * (1.0 + beta * (x - theta) / (1.0 + beta * jnp.abs(x - theta)))

def AbsSigmoidWrapper(beta, theta):
    return lambda x: AbsSigmoid(x, beta, theta)

@jax.jit
def ExpSigmoid(x, beta, theta):
    return 1.0 / (1.0 + jnp.exp(-beta * (x - theta)))

def ExpSigmoidWrapper(beta, theta):
    return lambda x: ExpSigmoid(x, beta, theta)

@jax.jit
def HeavySideSigmoid(x, beta, theta):
    return jnp.where(x < theta, 0.0, 1.0)

def HeavySideSigmoidWrapper(beta, theta):
    return lambda x: HeavySideSigmoid(x, beta, theta)

@jax.jit
def LinearSigmoid(x, beta, theta):
    return x * beta - theta

def LinearSigmoidWrapper(beta, theta):
    return lambda x: LinearSigmoid(x, beta, theta)

@jax.jit
def SemiLinearSigmoid(x, beta, theta):
    return jnp.where(x < theta, 0.0, x * beta - theta)

def SemiLinearSigmoidWrapper(beta, theta):
    return lambda x: SemiLinearSigmoid(x, beta, theta)

@jax.jit
def LogarithmicSigmoid(x, beta, theta):
    return jnp.log( beta * x - theta )

def LogarithmicSigmoidWrapper(beta, theta): 
    return lambda x: LogarithmicSigmoid(x, beta, theta)
