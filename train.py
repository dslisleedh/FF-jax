import jax
import jax.numpy as jnp
import jax.lax as lax

from src.model import *
from src.preprocessing import *
from src.optimizers import *
from src.utils import *
from src.losses import *

from flax import linen as nn
from flax.training.train_state import TrainState
from flax.metrics import tensorboard

import gin
import hydra
from hydra.utils import *
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
import logging

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import numpy as np


model_logger = logging.getLogger('Model')


@gin.configurable
def train(
        model: nn.Module, datasets: Sequence[tf.data.Dataset], seed: int,
        epochs: int, early_stopping: int
):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    train_ds, valid_ds, test_ds = datasets

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    params = model.initialize(jnp.ones((16, 28*28)), init_rng)

    for e in range(epochs):
        with tqdm(
                train_ds.as_numpy_iterator(), desc=f"Epoch {e+1}/{epochs} Training ...", colour='cyan',
                total=train_ds.cardinality().numpy()
        ) as pbar:
            epoch_train_loss = []
            for batch in pbar:
                x, y, sign = batch
                x = jnp.array(x, dtype=jnp.float32)
                y = jnp.array(y, dtype=jnp.float32)
                sign = jnp.array(sign, dtype=jnp.float32)
                loss, params = model.train_step(x, y, sign, params)
                epoch_train_loss.append(loss)
                pbar.set_postfix(loss=sum(epoch_train_loss) / len(epoch_train_loss))

        with tqdm(
            valid_ds.as_numpy_iterator(), desc=f"Epoch {e+1}/{epochs} Validation ...", colour='red',
            total=valid_ds.cardinality().numpy()
        ) as pbar2:
            y_true = []
            y_pred = []
            for batch in pbar2:
                x, y = batch
                x = jnp.array(x, dtype=jnp.float32)
                y = jnp.array(y, dtype=jnp.float32)
                y_true.append(y)
                y_pred.append(model.inference(x, params))

            y_true = jnp.concatenate(y_true)
            y_pred = jnp.concatenate(y_pred)
            acc = sum(y_true == y_pred) / len(y_true)
            model_logger.info(f'Epoch {e+1}/{epochs} Validation Result ... ACC: {acc * 100}')

    return None


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(main_config):
    import_external_configures()
    gin.parse_config_file(get_original_cwd() + '/config/hparams.gin')
    with open('./config.gin', 'w') as f:
        f.write(gin.operative_config_str())

    train()

if __name__ == '__main__':
    main()
