import gin

from functools import partial

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds


def preprocess(
        x: tf.Tensor, y: tf.Tensor, n_labels: int, augment: bool = True,
        pos_percentage: float = 0.5, flatten: bool = True
):
    x = tf.cast(x, tf.float32)
    if augment:
        crop_size = tf.random.uniform([], minval=26, maxval=29, dtype=tf.int32)
        x = tf.image.random_crop(x, (crop_size, crop_size, 1))
        x = tf.image.resize(x, (28, 28))
    x = x / 255.
    if flatten:
        x = tf.reshape(x, (-1,))
    y = tf.cast(y, tf.float32)

    if augment:
        # generate negative sample randomly
        if 1. - pos_percentage < tf.random.uniform([]):
            y = y + tf.cast(tf.random.uniform([], minval=1, maxval=n_labels, dtype=tf.int64), tf.float32)
            y = y % n_labels
            sign = tf.ones(())
        else:
            sign = tf.ones(()) * -1.
        return x, y, sign
    else:
        return x, y


@gin.configurable
def get_datasets(
        name: str = 'mnist', batch_size: int = 512, valid_percentage: int = 20, n_labels: int = 10,
        pos_percentage: float = 0.5, flatten: bool = True
):
    train_ds, valid_ds, test_ds = tfds.load(
        name, split=[
            f'train[:{int(100 - valid_percentage)}%]', f'train[{int(100 - valid_percentage)}%:]', 'test'],
        as_supervised=True
    )
    augment_func = partial(
        preprocess, n_labels=n_labels, augment=True, pos_percentage=pos_percentage,
        flatten=flatten
    )
    nonaugment_func = partial(preprocess, n_labels=n_labels, augment=False, flatten=flatten)
    train_ds = train_ds.map(augment_func).shuffle(10000).batch(batch_size, drop_remainder=True)\
        .prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(nonaugment_func).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(nonaugment_func).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds
