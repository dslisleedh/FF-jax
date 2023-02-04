import jax
import jax.numpy as jnp
from jax import lax

import gin


@gin.configurable
def simple_loss(goodness: jnp.ndarray, sign: jnp.ndarray, theta: float):
    # sign: -1 for positive samples, +1 for negative samples. To optimize pos/neg sample at the same time.
    goodness = goodness.reshape(goodness.shape[0], -1)
    return jnp.mean(
        jnp.sum((jnp.square(goodness) - theta) * sign, axis=1)
    )


@gin.configurable
def softplus_loss(goodness: jnp.ndarray, sign: jnp.ndarray, theta: float):
    # Loss from https://keras.io/examples/vision/forwardforward/
    goodness = goodness.reshape(goodness.shape[0], -1)
    logit = (goodness - theta) * sign
    return jnp.mean(
        jnp.log(1. + jnp.exp(logit))
    )
