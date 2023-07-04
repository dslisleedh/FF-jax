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

    @staticmethod
    def forward(fn, eps: float = 1e-8):
        def div_by_norm(*args, **kwargs):
            """
            In the paper, they layer normalized the active vector
            using simple normalization not subtracting mean and dividing std.

            if Dense:
                (B, N) / (B, 1)
            if Conv2D:
                (B, H, W, C) / (B, H, W, 1)
            """
            x = fn(*args, **kwargs)
            x_div = jnp.linalg.norm(x, axis=-1, ord=2, keepdims=True)
            normalized = x / (x_div + eps)
            goodness = jnp.sum(jnp.square(x), axis=-1)
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


@gin.configurable
class Conv2D(Layer):
    def __init__(
            self, n_filters: int, kernel_size: int | Sequence[int], init_func: callable, use_bias: bool,
            optimizer: callable, padding: str = 'SAME'
    ):
        super(Conv2D, self).__init__(init_func, use_bias, optimizer)
        self.n_filters = n_filters
        self.kernel_size = kernel_size if type(kernel_size) != int else (kernel_size, kernel_size)
        assert padding.upper() in ['SAME', 'VALID'], 'Padding must be either SAME or VALID'
        self.padding = padding.upper()

    def initialize(self, x: jnp.ndarray, rng: jax.random.PRNGKey):
        shape = self.kernel_size + (x.shape[-1], self.n_filters)
        params = super()._initialize(rng, shape)
        if self.use_bias:
            params[1] = params[1].reshape(1, 1, 1, -1)
        return self(x, *params), params

    @Layer.forward
    def __call__(self, x: jnp.ndarray, w: jnp.ndarray, b: Optional[jnp.ndarray] = None):
        x = lax.conv_general_dilated(
            x, w, (1, 1), self.padding, dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
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

        x = jnp.concatenate([x, y], axis=-1)  # In paper, they replace first 10 pixels with label.

        for layer, param in zip(self.layers, params):
            def loss_fn(p):
                x_normalized, goodness = layer(lax.stop_gradient(x), *p)
                loss = self.loss_fn(goodness, sign)
                return loss, x_normalized

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, x), grad = grad_fn(param)
            grads.append(grad)
            _loss += loss
        return _loss, grads

    def _update_gradients(self, params: Sequence[jnp.ndarray], grads: Sequence[jnp.ndarray]):
        params_updated = []
        for layer, param, grad in zip(self.layers, params, grads):
            param_ = layer.optimize(param, grad)
            params_updated.append(param_)
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
            x, goodness = layer(x, *param)
            goodness_total += goodness
        return goodness_total
