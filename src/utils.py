import gin
import jax.nn.initializers


def import_external_configures():
    def _register_initializers(module):
        gin.config.external_configurable(module, module='jax.nn.initializers')
    _register_initializers(jax.nn.initializers.lecun_normal)
    _register_initializers(jax.nn.initializers.glorot_normal)
    _register_initializers(jax.nn.initializers.he_normal)
