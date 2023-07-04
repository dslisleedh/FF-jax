import jax
import jax.numpy as jnp
from jax import lax

from typing import Optional, Sequence, Any, Tuple
from abc import abstractmethod

import gin


Pytree = Any


class Optimizer:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, grads: jnp.ndarray, state: Pytree):
        pass

    def __call__(self, params: jnp.ndarray, grads: jnp.ndarray, state: Pytree) -> Sequence[Tuple[jnp.ndarray, Pytree]]:
        update_val, new_state = self.update(grads, state)
        return params + update_val, new_state


@gin.configurable
class SGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3):
        self.learning_rate = learning_rate

    def initialize(self, params: jnp.ndarray) -> Pytree:
        state = {
            'learning_rate': jnp.asarray(self.learning_rate)
        }
        return state

    def update(self, grads: jnp.ndarray, state) -> Sequence[Tuple[jnp.ndarray, Pytree]]:
        lr = state['learning_rate']
        return -lr * grads, state


@gin.configurable
class MomentumSGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def initialize(self, params: jnp.ndarray) -> Pytree:
        state = {
            'learning_rate': jnp.asarray(self.learning_rate),
            'momentum': jnp.asarray(self.momentum),
            'velocity': jnp.zeros_like(params)
        }
        return state

    def update(self, grads: jnp.ndarray, state: Pytree) -> Sequence[Tuple[jnp.ndarray, Pytree]]:
        lr = state['learning_rate']
        momentum = state['momentum']
        velocity = state['velocity']

        velocity = momentum * velocity - lr * grads
        state['velocity'] = velocity
        return velocity, state


@gin.configurable
class NesterovMomentumSGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def initialize(self, params: jnp.ndarray) -> Pytree:
        state = {
            'learning_rate': jnp.asarray(self.learning_rate),
            'momentum': jnp.asarray(self.momentum),
            'velocity': jnp.zeros_like(params)
        }
        return state

    def update(self, grads: jnp.ndarray, state: Pytree) -> Sequence[Tuple[jnp.ndarray, Pytree]]:
        lr = state['learning_rate']
        momentum = state['momentum']
        velocity = state['velocity']

        velocity = self.momentum * velocity - lr * grads
        state['velocity'] = velocity
        return momentum * velocity - (lr * grads), state


@gin.configurable
class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def initialize(self, params: jnp.ndarray) -> Pytree:
        state = {
            'learning_rate': jnp.asarray(self.learning_rate),
            'epsilon': jnp.asarray(self.epsilon),
            'g_acc': jnp.zeros_like(params)
        }
        return state

    def update(self, grads: jnp.ndarray, state: Pytree) -> Sequence[Tuple[jnp.ndarray, Pytree]]:
        lr = state['learning_rate']
        epsilon = state['epsilon']
        g_acc = state['g_acc']

        g_acc += grads ** 2
        state['g_acc'] = g_acc
        return -lr / (epsilon + jnp.sqrt(g_acc)) * grads, state


@gin.configurable
class RMSProp(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-8, rho: float = 0.9):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rho = rho

    def initialize(self, params: jnp.ndarray) -> Pytree:
        state = {
            'learning_rate': jnp.asarray(self.learning_rate),
            'epsilon': jnp.asarray(self.epsilon),
            'rho': jnp.asarray(self.rho),
            'g_acc': jnp.zeros_like(params)
        }
        return state

    def update(self, grads: jnp.ndarray, state: Pytree) -> Sequence[Tuple[jnp.ndarray, Pytree]]:
        lr = state['learning_rate']
        epsilon = state['epsilon']
        rho = state['rho']
        g_acc = state['g_acc']

        g_acc = rho * g_acc + (1. - rho) * grads ** 2
        state['g_acc'] = g_acc
        return -lr / (epsilon + jnp.sqrt(g_acc)) * grads, state


@gin.configurable
class Adam(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-8, beta1: float = 0.9, beta2: float = 0.999):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def initialize(self, params: jnp.ndarray) -> Pytree:
        state = {
            'learning_rate': jnp.asarray(self.learning_rate),
            'epsilon': jnp.asarray(self.epsilon),
            'beta1': jnp.asarray(self.beta1),
            'beta2': jnp.asarray(self.beta2),
            'm': jnp.zeros_like(params),
            'v': jnp.zeros_like(params),
            'beta1_div': jnp.ones(()),
            'beta2_div': jnp.ones(())
        }
        return state

    def update(self, grads: jnp.ndarray, state: Pytree) -> Sequence[Tuple[jnp.ndarray, Pytree]]:
        lr = state['learning_rate']
        epsilon = state['epsilon']
        beta1 = state['beta1']
        beta2 = state['beta2']
        m = state['m']
        v = state['v']
        beta1_div = state['beta1_div']
        beta2_div = state['beta2_div']

        beta1_div *= beta1
        beta2_div *= beta2

        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * grads ** 2

        state['beta1_div'] = beta1_div
        state['beta2_div'] = beta2_div
        state['m'] = m
        state['v'] = v

        m_hat = m / (1. - beta1_div)
        v_hat = v / (1. - beta2_div)
        return -lr / (epsilon + jnp.sqrt(v_hat)) * m_hat, state


@gin.configurable
class AdaBelief(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, epsilon: float = 1e-16, beta1: float = 0.9, beta2: float = 0.999):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def initialize(self, params: jnp.ndarray) -> Pytree:
        state = {
            'learning_rate': jnp.asarray(self.learning_rate),
            'epsilon': jnp.asarray(self.epsilon),
            'beta1': jnp.asarray(self.beta1),
            'beta2': jnp.asarray(self.beta2),
            'm': jnp.zeros_like(params),
            's': jnp.zeros_like(params),
            'beta1_div': jnp.ones(()),
            'beta2_div': jnp.ones(())
        }
        return state

    def update(self, grads: jnp.ndarray, state: Pytree) -> Sequence[Tuple[jnp.ndarray, Pytree]]:
        lr = state['learning_rate']
        epsilon = state['epsilon']
        beta1 = state['beta1']
        beta2 = state['beta2']
        m = state['m']
        s = state['s']
        beta1_div = state['beta1_div']
        beta2_div = state['beta2_div']

        beta1_div *= beta1
        beta2_div *= beta2

        m = beta1 * m + (1 - beta1) * grads
        s = beta2 * s + (1 - beta2) * (grads - m) ** 2 + epsilon

        state['beta1_div'] = beta1_div
        state['beta2_div'] = beta2_div
        state['m'] = m
        state['s'] = s

        m_hat = m / (1. - beta1_div)
        s_hat = s / (1. - beta2_div)
        return -lr * m_hat / (jnp.sqrt(s_hat) + epsilon), state
