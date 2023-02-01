import jax
import jax.numpy as jnp
from jax import lax

import gin


# @gin.configurable
def loss(goodness: jnp.ndarray, sign: jnp.ndarray, theta: float):
    # sign: -1 for positive samples, +1 for negative samples. To optimize pos/neg sample at the same time.
    assert goodness.shape[0] == sign.shape[0]
    goodness = goodness.reshape(goodness.shape[0], -1)
    return jnp.mean(
        jnp.sum((jnp.square(goodness) - theta) * sign, axis=1)
    )
