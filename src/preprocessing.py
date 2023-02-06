import gin

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds


def preprocess(
        x: tf.Tensor, y: tf.Tensor, n_labels: int, augment: bool = True, pos_percentage: float = 0.5
):
    x = tf.cast(tf.reshape(x, (-1,)), tf.float32) / 125.5 - 1.
    y = tf.cast(y, tf.float32)

    if augment:
        # generate negative sample randomly
        if 1. - pos_percentage < tf.random.uniform([]):
            y = y + tf.cast(tf.random.uniform([], maxval=n_labels - 1, dtype=tf.int64), tf.float32) + tf.ones(())
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
        pos_percentage: float = 0.5
):
    train_ds, valid_ds, test_ds = tfds.load(
        name, split=[
            f'train[:{int(100 - valid_percentage)}%]', f'train[{int(100 - valid_percentage)}%:]', 'test'],
        as_supervised=True
    )
    train_ds = train_ds.map(lambda x, y: preprocess(x, tf.cast(y, tf.float32), n_labels, pos_percentage)).\
        batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(lambda x, y: preprocess(x, y, n_labels, augment=False))\
        .batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: preprocess(x, y, n_labels, augment=False))\
        .batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds
