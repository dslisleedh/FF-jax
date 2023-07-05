import gin

from functools import partial

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds


def preprocess(
        x: tf.Tensor, y: tf.Tensor, n_labels: int, augment: bool = True, flatten: bool = True,
):
    """
    1. Crop the image randomly to 26-28 pixels
    2. Resize the image to 28x28
    3. Normalize the image to [0, 1]
    4. Flatten the image

    5-1. Generate negative sample randomly
    5-2. Generate pairs of negative samples.
    """
    x = tf.cast(x, tf.float32) / 255.
    y = tf.cast(y, tf.float32)

    if augment:
        crop_h = tf.random.uniform([], minval=25, maxval=28, dtype=tf.int32)
        crop_w = tf.random.uniform([], minval=25, maxval=28, dtype=tf.int32)
        x = tf.image.random_crop(x, (tf.shape(x)[0], crop_h, crop_w, 1))
        x = tf.image.resize(x, (28, 28))

        pos_x = x
        neg_x = x
        pos_y = y
        neg_y = y + tf.cast(tf.random.uniform(
            (tf.shape(y)[0],), minval=1, maxval=n_labels, dtype=tf.int64), tf.float32)
        neg_y = neg_y % n_labels
        pos_sign = tf.ones((tf.shape(y)[0],), dtype=tf.float32) * -1.
        neg_sign = tf.ones((tf.shape(y)[0],), dtype=tf.float32)

        x = tf.concat([pos_x, neg_x], axis=0)
        y = tf.concat([pos_y, neg_y], axis=0)
        sign = tf.concat([pos_sign, neg_sign], axis=0)
        # state = tf.random.uniform(shape=tf.shape(y)[0]) < pos_percentage
        # y = tf.where(
        #     state, y, y + tf.cast(tf.random.uniform(tf.shape(y)[0], minval=1, maxval=n_labels, dtype=tf.int64), tf.float32))
        # y = y % n_labels
        # sign = tf.where(state, tf.ones_like(state, dtype=tf.float32), tf.ones_like(state, dtype=tf.float32) * -1.)
        if flatten:
            x = tf.reshape(x, (tf.shape(x)[0], -1))
        return x, y, sign

    if flatten:
        x = tf.reshape(x, (tf.shape(x)[0], -1))

    return x, y


@gin.configurable
def get_datasets(
        name: str = 'mnist', batch_size: int = 512, valid_percentage: int = 20, n_labels: int = 10,
        flatten: bool = True,
):
    train_ds, valid_ds, test_ds = tfds.load(
        name, split=[
            f'train[:{int(100 - valid_percentage)}%]', f'train[{int(100 - valid_percentage)}%:]', 'test'],
        as_supervised=True
    )
    augment_func = partial(preprocess, n_labels=n_labels, augment=True, flatten=flatten,)
    nonaugment_func = partial(preprocess, n_labels=n_labels, augment=False, flatten=flatten)
    train_ds = train_ds.shuffle(10000).batch(batch_size, drop_remainder=True).map(augment_func)\
        .prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(batch_size, drop_remainder=False).map(nonaugment_func).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=False).map(nonaugment_func).cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds
