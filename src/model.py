import jax
import jax.numpy as jnp
from jax import lax

from typing import Sequence, Optional
from copy import deepcopy
from functools import partial

import gin


class Layer:
    def __init__(self, init_func: callable, use_bias: bool, optimizer: callable):
        self.init_func = init_func()
        self.use_bias = use_bias
        self.optimizers = [deepcopy(optimizer()) for _ in range(2 if use_bias else 1)]

    def optimize(self, params: Sequence[jnp.ndarray], grads: Sequence[jnp.ndarray]):
        update_params = []
        for optimizer, param, grad in zip(self.optimizers, params, grads):
            update_param = optimizer(param, grad)
            update_params.append(update_param)
        return update_params

    def _initialize(self, rng: jax.random.PRNGKey, shape: Sequence):
        return_val = [self.init_func(rng, shape, dtype=jnp.float32)]
        if self.use_bias:
            b = jnp.zeros(shape[-1])
            return_val += [b]
        return return_val

    @staticmethod  # For further implementation like Convolution Layer
    def forward(fn):
        def div_by_norm(*args, **kwargs):
            x = fn(*args, **kwargs)
            shape = x.shape
            x_flatten = x.reshape(shape[0], -1)
            x_div = jnp.linalg.norm(x_flatten, axis=-1, ord=2)
            x_div = jnp.reshape(x_div, (shape[0], *(1 for _ in shape[1:])))
            normalized = x / (x_div + 1e-8)
            goodness = jnp.sum(jnp.square(x_flatten), axis=-1)
            return normalized, goodness
        return div_by_norm


@gin.configurable
class Dense(Layer):
    def __init__(self, n_units: int, init_func: callable, use_bias: bool, optimizer: callable):
        super().__init__(init_func, use_bias, optimizer)
        self.n_units = n_units

    def initialize(self, x: jnp.ndarray, rng: jax.random.PRNGKey):
        shape = (x.shape[-1], self.n_units)
        params = super()._initialize(rng, shape)
        return self(x, *params), params

    @Layer.forward
    def __call__(self, x: jnp.ndarray, w: jnp.ndarray, b: Optional[jnp.ndarray] = None):
        x = jnp.dot(x, w)
        if self.use_bias:
            x = x + b
        x = jax.nn.relu(x)
        return x


# @gin.configurable
# class UnsupervisedModel:  # Will implement this later. HOW TO MAKE HYBRID SAMPLE?
#     def __init__(self, config):
#         raise NotImplementedError


@gin.configurable
class SupervisedModel:
    def __init__(
            self, n_layers: int, loss_fn: callable, theta: float,
            n_labels: int, layer: Layer
    ):
        self.n_layers = n_layers
        self.loss_fn = partial(loss_fn, theta=theta)
        self.layer = layer
        self.layers = []
        self.n_labels = n_labels

    def initialize(self, x: jnp.ndarray, rng: jax.random.PRNGKey):
        rngs = jax.random.split(rng, self.n_layers)
        pseudo_y = jnp.ones((x.shape[0], self.n_labels))
        x = jnp.concatenate([x, pseudo_y], axis=-1)

        params = []
        for i in range(self.n_layers):
            layer = self.layer()
            (x, _), param = layer.initialize(x, rngs[i])
            self.layers.append(layer)
            params.append(param)
        return params

    @partial(jax.jit, static_argnums=(0,))
    def _get_gradients(
            self, x: jnp.ndarray, y: jnp.ndarray, sign: jnp.ndarray, params: Sequence
    ):
        grads = []
        _loss = jnp.zeros(())

        x = jnp.concatenate([x, y], axis=-1)

        for layer, param in zip(self.layers, params):
            def loss_fn(params):
                x_normalized, goodness = layer(x, *params)
                loss = self.loss_fn(goodness / layer.n_units, sign)
                return loss, x_normalized

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, x), grad = grad_fn(param)
            grads.append(grad)
            _loss += loss
        return _loss, grads

    def _update_gradients(self, params: Sequence[jnp.ndarray], grads: Sequence[jnp.ndarray]):
        # Later: Think about gradient updates to be vectorized?
        params_updated = []
        for layer, param, grad in zip(self.layers, params, grads):
            params = layer.optimize(param, grad)
            params_updated.append(params)
        return params_updated

    def train_step(self, x: jnp.ndarray, y: jnp.ndarray, sign: jnp.ndarray, params: Sequence[jnp.ndarray]):
        y = jax.nn.one_hot(y, num_classes=self.n_labels)
        loss, grads = self._get_gradients(x, y, sign, params)
        params_updated = self._update_gradients(params, grads)
        return loss, params_updated

    def _return_label(self, i: int, n_samples: int):
        return jax.nn.one_hot(jnp.ones((n_samples,)) * i, self.n_labels)

    def inference(self, x: jnp.ndarray, params: Sequence[jnp.ndarray]):
        label_fn = partial(self._return_label, n_samples=x.shape[0])
        labels = jax.vmap(label_fn)(jnp.arange(self.n_labels))
        inference = partial(self.__call__, x=x, params=params)
        goodness_labels = jax.vmap(inference)(y=labels)
        goodness_labels = jnp.argmax(goodness_labels, axis=0)
        return goodness_labels

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, params: Sequence[jnp.ndarray]):
        # Return goodness of given label
        goodness_total = jnp.zeros((x.shape[0],))
        x = jnp.concatenate([x, y], axis=-1)
        for layer, param in zip(self.layers, params):
            # x = jnp.concatenate([x, y], axis=-1) Concatenate x, y for every layers like diffusion? of just input?
            x, goodness = layer(x, *param)
            goodness_total += goodness
        return goodness_total
