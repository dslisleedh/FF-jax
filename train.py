import jax
import jax.numpy as jnp
import jax.lax as lax

# To prevent cuda-related errors when import jax and tensorflow in the same file
a = jnp.ones((16, 256, 256, 3))
del a
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from src.model import *
from src.preprocessing import *
from src.optimizers import *
from src.utils import *
from src.losses import *

import gin
import hydra
from hydra.utils import *
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
import logging

import tensorflow_datasets as tfds
from tensorboardX import SummaryWriter
import numpy as np
import pickle


model_logger = logging.getLogger('Model')


@gin.configurable
def train(
        model: callable, datasets: Sequence, seed: int,
        epochs: int, early_stopping: EarlyStopping
):
    with open('./hparams.gin', 'w') as f:
        f.write(gin.operative_config_str())

    tensorboard_logger = tf.summary.create_file_writer('./logs')

    tf.random.set_seed(seed)
    np.random.seed(seed)
    train_ds, valid_ds, test_ds = datasets

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    params, opt_state = model.initialize(jnp.ones((16, 28*28)), init_rng)

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
                loss, params, opt_state = model.train_step(x, y, sign, params, opt_state)
                epoch_train_loss.append(loss)
                pbar.set_postfix(loss=sum(epoch_train_loss) / len(epoch_train_loss))
                with tensorboard_logger.as_default():
                    tf.summary.scalar('train_loss', sum(epoch_train_loss) / len(epoch_train_loss), step=e)

        with tqdm(
            valid_ds.as_numpy_iterator(), desc=f"Epoch {e+1}/{epochs} Validating ...", colour='red',
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
            acc = (sum(y_true == y_pred) / len(y_true)) * 100.
            model_logger.info(f'Epoch {e+1}/{epochs} Validation Result ... ACC: {acc:.4f}%')
            with tensorboard_logger.as_default():
                tf.summary.scalar('valid_acc', acc, step=e)
            early_stopping(acc, params, opt_state)
            if early_stopping.is_stop:
                break

    best_params = early_stopping.best_params
    best_opt_state = early_stopping.best_opt_state
    with open('./best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    with open('./best_opt_state.pkl', 'wb') as f:
        pickle.dump(best_opt_state, f)

    with tqdm(
        test_ds.as_numpy_iterator(), desc=f"Testing ...", colour='green', total=test_ds.cardinality().numpy()
    ) as pbar3:
        y_true = []
        y_pred = []
        for batch in pbar3:
            x, y = batch
            x = jnp.array(x, dtype=jnp.float32)
            y = jnp.array(y, dtype=jnp.float32)
            y_true.append(y)
            y_pred.append(model.inference(x, best_params))

        y_true = jnp.concatenate(y_true)
        y_pred = jnp.concatenate(y_pred)
        test_acc = (sum(y_true == y_pred) / len(y_true)) * 100.
        model_logger.info(f'Test Result ... ACC: {test_acc:.4f}%')


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(main_config):
    import_external_configures()
    gin.parse_config_file(get_original_cwd() + '/config/hparams.gin')

    train()


if __name__ == '__main__':
    main()
