import jax
import jax.numpy as jnp
from jax import lax

import gin


@gin.configurable
def simple_loss(goodness: jnp.ndarray, sign: jnp.ndarray, theta: float):
    # sign: -1 for positive samples, +1 for negative samples. To optimize pos/neg sample at the same time.
    goodness = goodness.reshape(goodness.shape[0], -1)
    return jnp.mean((goodness - theta) * sign)


@gin.configurable
def softplus_loss(goodness: jnp.ndarray, sign: jnp.ndarray, theta: float):
    # Loss from https://keras.io/examples/vision/forwardforward/
    goodness = goodness.reshape(goodness.shape[0], -1)
    logit = (goodness - theta) * sign
    return jnp.mean(
        jnp.log(1. + jnp.exp(logit))
    )


@gin.configurable
def probabilistic_loss(goodness: jnp.ndarray, sign: jnp.ndarray, theta: float):
    # Loss from https://github.com/nebuly-ai/nebullvm/blob/main/apps/accelerate/forward_forward/forward_forward/utils/modules.py
    prob = jax.nn.sigmoid(goodness - theta)
    loss = jnp.log(prob + 1e-6) * sign
    return jnp.mean(loss)
