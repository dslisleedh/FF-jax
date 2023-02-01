import jax
import jax.numpy as jnp
from jax import lax

from typing import Optional

import gin


# @gin.configurable
class SGD:
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.learning_rate = learning_rate

    def __call__(self, grads: jnp.ndarray):
        return -self.learning_rate * grads


# @gin.configurable
class RMSProp:
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-8, rho: float = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rho = rho
        self.g_acc = 0.

    def __call__(self, grads: jnp.ndarray):
        self.g_acc = self.rho * self.g_acc + (1. - self.rho) * jnp.square(grads)
        return -self.learning_rate * grads / (jnp.sqrt(self.g_acc) + self.epsilon)


# @gin.configurable
class Adam:
    def __init__(
            self, learning_rate: float = 1e-3, epsilon: float = 1e-8,
            beta1: float = 0.9, beta2: float = 0.999
    ):
        super(Adam, self).__init__()
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_acc = 0.
        self.v_acc = 0.
        self.beta1_div = 1.
        self.beta2_div = 1.

    def __call__(self, grads: jnp.ndarray):
        self.m_acc = self.beta1 * self.m_acc + (1. - self.beta1) * grads
        self.v_acc = self.beta2 * self.v_acc + (1. - self.beta2) * jnp.square(grads)
        self.beta1_div *= self.beta1
        self.beta2_div *= self.beta2
        m_hat = self.m_acc / (1. - self.beta1_div)
        v_hat = self.v_acc / (1. - self.beta2_div)
        return -self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.epsilon)


# @gin.configurable
class AdaBelief:
    def __init__(
            self, learning_rate: float = 1e-3, epsilon: float = 1e-16,
            beta1: float = 0.9, beta2: float = 0.999
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_acc = 0.
        self.v_acc = 0.
        self.beta1_div = 1.
        self.beta2_div = 1.

    def __call__(self, grads: jnp.ndarray):
        self.m_acc = self.beta1 * self.m_acc + (1. - self.beta1) * grads
        self.v_acc = self.beta2 * self.v_acc + (1. - self.beta2) * jnp.square(grads - self.m_acc)
        self.beta1_div *= self.beta1
        self.beta2_div *= self.beta2
        m_hat = self.m_acc / (1. - self.beta1_div)
        v_hat = self.v_acc / (1. - self.beta2_div)
        return -self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
