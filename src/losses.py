import jax
import jax.numpy as jnp
from jax import lax, Array

import gin


@gin.configurable
def mse_loss(goodness: jnp.ndarray, sign: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = jax.nn.sigmoid((goodness - theta) * sign)
    return jnp.mean(jnp.square(p))


@gin.configurable
def softplus_loss(goodness: jnp.ndarray, sign: jnp.ndarray, theta: float) -> jnp.ndarray:
    # Loss from https://keras.io/examples/vision/forwardforward/
    logit = (goodness - theta) * sign
    return jnp.mean(
        jnp.log(1. + jnp.exp(logit))
    )


@gin.configurable
def probabilistic_loss(goodness: jnp.ndarray, sign: jnp.ndarray, theta: float) -> jnp.ndarray:
    # Loss from https://github.com/nebuly-ai/nebullvm/blob/main/apps/accelerate/forward_forward/forward_forward/utils/modules.py
    prob = jax.nn.sigmoid((goodness - theta) * sign)
    loss = -jnp.log(1. - prob + 1e-6)
    return jnp.mean(loss)


def _get_pos_sub_neg(goodness: jnp.ndarray) -> jnp.ndarray:
    split_idx = int(goodness.shape[0] / 2)
    pos = goodness[:split_idx]
    neg = goodness[split_idx:]
    return pos - neg


@gin.configurable
def simba_loss(goodness: jnp.ndarray, sign: jnp.ndarray, alpha: float) -> jnp.ndarray:
    sub_val = _get_pos_sub_neg(goodness)
    return jnp.mean(
        jnp.log(1. + jnp.exp(-alpha * sub_val))
    )


@gin.configurable
def swish_simba_loss(goodness: jnp.ndarray, sign: jnp.ndarray, alpha: float) -> jnp.ndarray:
    sub_val = _get_pos_sub_neg(goodness)
    return jnp.mean(
        (-alpha * sub_val) * jax.nn.sigmoid(-alpha * sub_val)
    )
