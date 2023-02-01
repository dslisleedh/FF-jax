import jax
import jax.numpy as jnp
from jax import lax

from typing import Optional

import gin


class Optimizer:
    def update(self, grads: jnp.ndarray):
        raise NotImplementedError

    def __call__(self, params: jnp.ndarray, grads: jnp.ndarray):
        update_val = self.update(grads)
        return params + update_val


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3):
        self.learning_rate = learning_rate

    def update(self, grads: jnp.ndarray):
        return -self.learning_rate * grads


class MomentumSGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0.

    def update(self, grads: jnp.ndarray):

        self.velocity = self.momentum * self.velocity - self.learning_rate * grads
        return self.velocity


class NesterovMomentumSGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0.

    def update(self, grads: jnp.ndarray):
        self.velocity = self.momentum * self.velocity - self.learning_rate * grads
        return self.momentum * self.velocity - (self.learning_rate * grads)


class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.g_acc = 0.

    def update(self, grads: jnp.ndarray):
        self.g_acc += grads ** 2
        return -self.learning_rate / (self.epsilon + jnp.sqrt(self.g_acc)) * grads


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-8, rho: float = 0.9):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rho = rho
        self.g_acc = 0.

    def update(self, grads: jnp.ndarray):
        self.g_acc = self.rho * self.g_acc + (1 - self.rho) * grads ** 2
        return -self.learning_rate / (self.epsilon + jnp.sqrt(self.g_acc)) * grads


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-8, beta1: float = 0.9, beta2: float = 0.999):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0.
        self.v = 0.
        self.beta1_div = 1.
        self.beta2_div = 1.

    def update(self, grads: jnp.ndarray):
        self.beta1_div *= self.beta1
        self.beta2_div *= self.beta2

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat = self.m / (1 - self.beta1_div)
        v_hat = self.v / (1 - self.beta2_div)
        return -self.learning_rate / (self.epsilon + jnp.sqrt(v_hat)) * m_hat


class AdaBelief(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-16, beta1: float = 0.9, beta2: float = 0.999):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0.
        self.s = 0.
        self.beta1_div = 1.
        self.beta2_div = 1.

    def update(self, grads: jnp.ndarray):
        self.beta1_div = self.beta1_div * self.beta1
        self.beta2_div = self.beta2_div * self.beta2

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.s = self.beta2 * self.s + (1 - self.beta2) * (grads - self.m) ** 2 + self.epsilon
        m_hat = self.m / (1 - self.beta1_div)
        s_hat = self.s / (1 - self.beta2_div)
        return -self.learning_rate * m_hat / (jnp.sqrt(s_hat) + self.epsilon)
