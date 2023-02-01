import jax
import jax.numpy as jnp
from jax import lax

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds


# def random_negative_sample(x: tf.Tensor, y: tf.Tensor, n_labels: int):
#
#
#
# def get_dataset(name: str = 'mnist', batch_size: int = 512, valid_percentage: int = 20):
#     train_ds, valid_ds, test_ds = tfds.load(
#         name, split=[
#             f'train[:{int(100 - valid_percentage)}%]', f'train[{int(100 - valid_percentage)}%:]', 'test'],
#         as_supervised=True
#     )
