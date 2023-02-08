import gin

import jax.numpy as jnp
import jax.nn.initializers
import jax.numpy as jnp

import logging
from copy import deepcopy
from typing import Sequence


es_logger = logging.getLogger('EarlyStopping')


def import_external_configures():
    def _register_initializers(module):
        gin.config.external_configurable(module, module='jax.nn.initializers')
    _register_initializers(jax.nn.initializers.lecun_normal)
    _register_initializers(jax.nn.initializers.glorot_normal)
    _register_initializers(jax.nn.initializers.he_normal)
    _register_initializers(jax.nn.initializers.variance_scaling)


@gin.configurable
class EarlyStopping:
    def __init__(self, mode: str = 'min', patience: int = 10, verbose: bool = False):
        assert mode in ['min', 'max']
        self.mode = mode
        self.history = -jnp.inf if mode == 'max' else jnp.inf
        self.patience = patience
        self.epochs = 0
        self.count = 0
        self.verbose = verbose
        self.is_stop = False
        self.best_params = None
        self.best_epoch = 0

    def update(self, params, epoch):
        self.best_params = deepcopy(params)
        self.best_epoch = epoch
        self.count = 0

    def __call__(self, check_val: float, params: Sequence[jnp.ndarray]):
        self.epochs += 1
        if self.mode == 'max':
            if check_val > self.history:
                self.history = check_val
                self.update(params, self.epochs)
            else:
                self.count += 1
        else:
            if check_val < self.history:
                self.history = check_val
                self.update(params, self.epochs)
            else:
                self.count += 1
        if self.count >= self.patience:
            self.is_stop = True
            if self.verbose:
                es_logger.info(f'Early stopping at {self.epochs}th epoch')
                es_logger.info(f'Restore best params at {self.best_epoch}th epoch')
        else:
            if self.verbose:
                es_logger.info(f'Current patience: {self.count}/{self.patience} - Current best: {self.history}')
