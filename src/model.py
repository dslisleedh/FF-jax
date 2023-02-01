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
        params = [self.w]
        if self.use_bias:
            params += [self.b]
        return params

    def optimize(self, params_grad: Sequence):
        w = self.w_optimizer(params_grad[0])
        self.w += w
        if self.use_bias:
            b = self.b_optimizer(params_grad[1])
            self.b += b

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
        return self(x, *self.get_params())

    @Layer.forward
    def __call__(self, x: jnp.ndarray, w: jnp.ndarray, b: Optional[jnp.ndarray] = None):
        x = jnp.dot(x, w)
        if self.use_bias:
            x = x + b
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

    @partial(jax.jit, static_argnums=(0,))
    def train(self, x: jnp.ndarray, y: jnp.ndarray, sign: jnp.ndarray):
        total_loss = 0

        for layer in self.layers:
            x = jnp.concatenate([x, y], axis=-1)
            params = layer.get_params()

            @jax.jit
            def loss_fn(params):
                x_normalized, goodness = layer(x, *params)
                loss = self.loss_fn(goodness, sign)
                return loss, x_normalized

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, x), grad = grad_fn(params)
            layer.optimize(grad)
            total_loss += loss

        return total_loss

    def _inference(self, x: jnp.ndarray, y_num: int):  # For given y
        y = jnp.ones((x.shape[0])) * y_num
        y = jax.nn.one_hot(y, self.config.n_labels)
        x = jnp.concatenate([x, y], axis=-1)

        goodness_layer = jnp.zeros((len(self.layers),))
        for i, layer in enumerate(self.layers):
            x, _ = layer(x, layer.get_params())
            goodness_layer[i] = jnp.sum(jnp.square(x))
        return goodness_layer

    @partial(jax.jit, static_argnums=(0,))
    def inference(self, x: jnp.ndarray):
        goodness_label = []

        res = lax.fori_loop(0, self.config.n_labels, lambda i, _: self._inference(x, i), None)
        return res

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray):
        res = jnp.zeros((x.shape[0], 1))
        for layer in self.layers:
            x = jnp.concatenate([x, y], axis=-1)
            x, goodness = layer(x, layer.get_params())
            res += goodness
        return res
