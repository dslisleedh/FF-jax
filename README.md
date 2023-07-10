# FF-jax[[Paper]](https://arxiv.org/abs/2212.13345)

Unofficial implementation of Forward-Forward algorithm by jax.  


## Usage
```shell
# download script from git
git clone https://github.com/dslisleedh/FF-jax.git
cd FF-jax

# create environment
conda create --name <env> --file requirements.txt
conda activate <env>
# if this not working, install below packages manually  
# jax, jaxlib (https://github.com/google/jax#installation)  
# einops, tensorflow, tensorflow_datasets, tqdm, hydra-core, hydra-colorlog, omegaconf, gin-config  

# run ! 
python train.py
```

You can easily change train setting under ./config/hparams.gin  # config.yaml is for hydra that create and set working directory 


Hyperparameters
 - [Losses](https://github.com/dslisleedh/FF-jax/blob/master/src/losses.py)
   - mse_loss
   - softplus_loss (used in Original Paper)
   - probabilistic_loss
   - symba_loss
   - swish_symba_loss
 - [Optimizers](https://github.com/dslisleedh/FF-jax/blob/master/src/optimizers.py)
   - SGD
   - MomentumSGD
   - NesterovMomentumSGD
   - AdaGrad
   - RMSProp
   - Adam
   - AdaBelief
 - [Initializers](https://github.com/dslisleedh/FF-jax/blob/85d44df5a6a3ddf646229d9a84600e9b33735d32/src/utils.py#L15)
   - jax.nn.initializers.lecun_normal
   - jax.nn.initializers.glorot_normal
   - jax.nn.initializers.he_normal
   - jax.nn.initializers.variance_scaling
 - and others like n_layers, n_units, ...


## TODO
 - [ ] Add Local Conv With Peer Normalization

## What about...
 - add online training model?