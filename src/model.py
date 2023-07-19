import jax
import jax.numpy as jnp
import jax.lax as lax

from typing import Sequence, Optional, Any
from copy import deepcopy
from functools import partial

from abc import abstractmethod

import gin


Pytree = Any


def forward_layernorm(fn, eps: float = 1e-8) -> callable:
    def _layer_norm_fast(*args, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        In the paper, they layer normalized the active vector
        using simple normalization not subtracting mean and dividing std.

        if Dense:
            (B, N) / (B, 1)
        if Conv2D:
            (B, H, W, C) / (B, 1, 1, 1)
        """
        x = fn(*args, **kwargs)
        shape = x.shape
        reduce_dim = tuple(i for i in range(1, len(shape)))
        norm = jnp.linalg.norm(x, axis=reduce_dim, ord=2, keepdims=True)
        normalized = x / (norm + eps)
        goodness = jnp.sum(x, axis=reduce_dim)
        return normalized, goodness
    return _layer_norm_fast


class Layer:
    def __init__(self, init_func: callable, use_bias: bool, optimizer: callable):
        self.init_func = init_func()
        self.use_bias = use_bias
        self.optimizers = [deepcopy(optimizer()) for _ in range(2 if use_bias else 1)]

    def optimize(
            self, params: Sequence[jnp.ndarray], grads: Sequence[jnp.ndarray], opt_state: Sequence[Pytree]
    ) -> tuple[Sequence[jnp.ndarray], Sequence[Pytree]]:
        update_params = []
        new_states = []
        for optimizer, param, grad, state in zip(self.optimizers, params, grads, opt_state):
            update_param, new_state = optimizer(param, grad, state)
            update_params.append(update_param)
            new_states.append(new_state)
        return update_params, new_states

    def _initialize(self, rng: jax.random.PRNGKey, shape: Sequence) -> tuple[Sequence[jnp.ndarray], Sequence[Pytree]]:
        params = [self.init_func(rng, shape, dtype=jnp.float32)]  # w
        if self.use_bias:
            b = jnp.zeros(shape[-1])
            params += [b]

        opt_state = []
        for p, optimizer in zip(params, self.optimizers):
            opt_state.append(optimizer.initialize(p))

        return params, opt_state


@gin.configurable
class Dense(Layer):
    def __init__(self, n_units: int, init_func: callable, use_bias: bool, optimizer: callable):
        super().__init__(init_func, use_bias, optimizer)
        self.n_units = n_units

    def initialize(
            self, x: jnp.ndarray, rng: jax.random.PRNGKey
    ) -> tuple[jnp.ndarray, Sequence[jnp.ndarray], Sequence[Pytree]]:
        shape = (x.shape[-1], self.n_units)
        params, opt_state = super()._initialize(rng, shape)
        return self(x, *params), params, opt_state

    @forward_layernorm
    def __call__(self, x: jnp.ndarray, w: jnp.ndarray, b: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        x = jnp.dot(x, w)
        if self.use_bias:
            x = x + b
        x = jax.nn.relu(x)
        return x


# Remove this for now. Will implement later.
# @gin.configurable
# class Conv2D(Layer):
#     def __init__(
#             self, n_filters: int, kernel_size: int | Sequence[int], init_func: callable, use_bias: bool,
#             optimizer: callable, padding: str = 'SAME'
#     ):
#         super(Conv2D, self).__init__(init_func, use_bias, optimizer)
#         self.n_filters = n_filters
#         self.kernel_size = kernel_size if type(kernel_size) != int else (kernel_size, kernel_size)
#         assert padding.upper() in ['SAME', 'VALID'], 'Padding must be either SAME or VALID'
#         self.padding = padding.upper()
#
#     def initialize(self, x: jnp.ndarray, rng: jax.random.PRNGKey):
#         shape = self.kernel_size + (x.shape[-1], self.n_filters)
#         params = super()._initialize(rng, shape)
#         if self.use_bias:
#             params[1] = params[1].reshape(1, 1, 1, -1)
#         return self(x, *params), params
#
#     @Layer.forward
#     def __call__(self, x: jnp.ndarray, w: jnp.ndarray, b: Optional[jnp.ndarray] = None):
#         x = lax.conv_general_dilated(
#             x, w, (1, 1), self.padding, dimension_numbers=('NHWC', 'HWIO', 'NHWC')
#         )
#         if self.use_bias:
#             x = x + b
#         x = jax.nn.relu(x)
#         return x


class Model:
    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


@gin.configurable
class SupervisedModel(Model):
    def __init__(
            self, n_layers: int, loss_fn: callable, n_labels: int, layer: Layer
    ):
        self.n_layers = n_layers
        self.loss_fn = loss_fn
        self.layer = layer
        self.layers = []
        self.n_labels = n_labels
        # TODO: To get Lists of Layers. Than n_layers should be removed. And Loss_fns must be stored in each layer.

    def initialize(self, x: jnp.ndarray, rng: jax.random.PRNGKey) -> tuple[Sequence[jnp.ndarray], Sequence[Pytree]]:
        rngs = jax.random.split(rng, self.n_layers)
        pseudo_y = jnp.ones((x.shape[0], self.n_labels))
        x = jnp.concatenate([x, pseudo_y], axis=-1)

        params = []
        opt_states = []
        for i in range(self.n_layers):
            layer = deepcopy(self.layer())
            (x, _), param, opt_state = layer.initialize(x, rngs[i])
            self.layers.append(layer)
            params.append(param)
            opt_states.append(opt_state)
        return params, opt_states

    @partial(jax.jit, static_argnums=(0,))
    def train_step(
            self, x: jnp.ndarray, y: jnp.ndarray, sign: jnp.ndarray, params: Sequence[jnp.ndarray],
            opt_state: Sequence[Pytree]
    ) -> tuple[jnp.ndarray, Sequence[jnp.ndarray], Sequence[Pytree]]:
        y = jax.nn.one_hot(y, num_classes=self.n_labels)
        x = jnp.concatenate([x, y], axis=-1)  # In paper, they replace first 10 pixels with label.

        params_new = []
        state_new = []
        _loss = jnp.zeros(())

        for layer, param, state in zip(self.layers, params, opt_state):
            def loss_fn(p):
                x_normalized, goodness = layer(lax.stop_gradient(x), *p)
                loss = self.loss_fn(goodness, sign)
                return loss, x_normalized
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, x), grad = grad_fn(param)
            param_, state_ = layer.optimize(param, grad, state)
            params_new.append(param_)
            state_new.append(state_)
            _loss += loss
        return _loss, params_new, state_new

    def _return_label(self, i: jnp.ndarray | int, n_samples: jnp.ndarray | int) -> jnp.ndarray:
        return jax.nn.one_hot(jnp.ones((n_samples,)) * i, self.n_labels)

    def inference(self, x: jnp.ndarray, params: Sequence[jnp.ndarray]) -> jnp.ndarray:
        labels = jax.vmap(self._return_label, in_axes=(0, None))(jnp.arange(self.n_labels), x.shape[0])
        goodness_labels = jax.vmap(self.__call__, in_axes=(None, 0, None))(x, labels, params)
        goodness_labels = jnp.argmax(goodness_labels, axis=0)
        return goodness_labels

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, params: Sequence[jnp.ndarray]) -> jnp.ndarray:
        # Return goodness of given label
        goodness_total = jnp.zeros((x.shape[0],))
        x = jnp.concatenate([x, y], axis=-1)
        for layer, param in zip(self.layers, params):
            x, goodness = layer(x, *param)
            goodness_total += goodness
        return goodness_total
