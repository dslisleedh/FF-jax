# FF-jax

Unofficial implementation of Forward-Forward algorithm by jax.  
[Paper](https://arxiv.org/abs/2212.13345)  


## TODO
 - [X] Layer and Models
 - [X] Losses and Optimizers
 - [X] Preprocess codes for supervised model
 - [X] Train codes of supervised model
 - [X] Fix LayerNorm(To calculate norm at channel)
 - [X] Fix optimizer weights to be not stored in class for load
 - [ ] Add Local Conv With Peer Normalization
 - [ ] Add other losses


## What about...
 - change preprocessing codes to make pairs of positive sample and negative samples for simba loss?
 - add online training model?