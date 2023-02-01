import jax
import jax.numpy as jnp
from jax import lax

from jax.nn.initializers import glorot_normal

from typing import Sequence, Optional
from copy import deepcopy
from functools import partial

import gin


class Layer:
    def __init__(self, init_func: callable, use_bias: bool, optimizer: callable):
        self.init_func = init_func()
        self.use_bias = use_bias
        self.w_optimizer = optimizer()
        if self.use_bias:
            self.b_optimizer = deepcopy(optimizer())

    def get_params(self):
        weights = [self.w]
        if self.use_bias:
            weights.append(self.b)
        return weights

    def optimize(self, params_grad: Sequence):
        self.w += self.w_optimizer(params_grad[0])
        if self.use_bias:
            self.b += self.b_optimizer(params_grad[1])

    def _initialize(self, rng: jax.random.PRNGKey, shape: Sequence):
        self.w = self.init_func(rng, shape, dtype=jnp.float32)
        if self.use_bias:
            self.b = jnp.zeros(shape[-1])

    @staticmethod  # For further implementation like Convolution Layer
    def forward(fn):
        def div_by_norm(*args, **kwargs):
            x = fn(*args, **kwargs)
            shape = x.shape
            x_flatten = x.reshape(shape[0], -1)
            x_div = jnp.linalg.norm(x_flatten, axis=-1, ord=2)
            x_div = jnp.reshape(x_div, (shape[0], *(1 for _ in shape[1:])))
            normalized = x / x_div
            goodness = jnp.sum(jnp.square(x_flatten), axis=-1)
            return normalized, goodness
        return div_by_norm


# @gin.configurable
class Dense(Layer):
    def __init__(self, n_units: int, init_func: callable, use_bias: bool, optimizer: callable):
        super().__init__(init_func, use_bias, optimizer)
        self.n_units = n_units

    def initialize(self, x: jnp.ndarray, rng: jax.random.PRNGKey):
        shape = (x.shape[-1], self.n_units)
        super()._initialize(rng, shape)
        return self(x)

    @Layer.forward
    def __call__(self, x: jnp.ndarray):
        x = jnp.dot(x, self.w)
        if self.use_bias:
            x = x + self.b
        x = jax.nn.relu(x)
        return x


@gin.configurable
class UnsupervisedModel:  # Will implement this later. HOW TO MAKE HYBRID SAMPLE?
    def __init__(self, config):
        raise NotImplementedError


# @gin.configurable
class SupervisedModel:
    def __init__(self, config):
        self.config = config
        self.loss_fn = partial(self.config.loss_fn, theta=config.theta)
        self.layers = []

    def initialize(self, x: jnp.ndarray, rng: jax.random.PRNGKey):
        rngs = jax.random.split(rng, self.config.n_layers)
        layer_config = self.config.layer_config if self.config.layer_config is Sequence \
            else [self.config.layer_config] * self.config.n_layers
        pseudo_y = jnp.ones((x.shape[0], self.config.n_labels))

        for i, config in enumerate(layer_config):
            layer = self.config.layer(**config)
            x = jnp.concatenate([x, pseudo_y], axis=-1)
            x, _ = layer.initialize(x, rngs[i])
            self.layers.append(layer)

    @jax.jit
    def train(self, batch):
        x, y, sign = batch

        for layer in self.layers:
            x = jnp.concatenate([x, y], axis=-1)
            params = layer.get_params()

            def loss_fn(params):
                x_normalized, goodness = layer(x, params)
                loss = self.loss_fn(goodness, y, sign)
                return loss, x_normalized

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, x), grad = grad_fn(params)
            layer.optimize(grad)

    @jax.jit
    def inference(self, x: jnp.ndarray):
        goodness_label = []
        for i in range(self.config.n_labels):
            y = jnp.ones((x.shape[0], 1)) * i
            y = jax.nn.one_hot(y, self.config.n_labels)

            goodness_layer = []
            for layer in self.layers:
                x = jnp.concatenate([x, y], axis=-1)
                x, goodness = layer(x)
                goodness_layer.append(goodness)
            goodness_layer = jnp.concatenate(goodness_layer, axis=-1)
            goodness_label.append(goodness_layer)

        label = jnp.stack(goodness_label, axis=-1).sum(axis=-1).argmax(axis=-1)
        return label
