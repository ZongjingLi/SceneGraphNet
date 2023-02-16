# Physical Scene Graph Network
This repo is trying to reproduce the project "Learning Physical Graph Representations from Visual Scenes"
[[Paper]](https://arxiv.org/pdf/1904.11694.pdf)
[[Project Page]](https://sites.google.com/view/neural-logic-machines)

## Usage

Jax version:

```py
from einshape import jax_einshape as einshape
from jax import numpy as jnp

a = jnp.array([[1, 2], [3, 4]])
b = einshape("ij->(ij)", a)
# b is [1, 2, 3, 4]
```