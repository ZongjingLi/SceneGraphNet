# Physical Scene Graph Network
This repo is trying to reproduce the project "Learning Physical Graph Representations from Visual Scenes"
[[Paper]](https://arxiv.org/pdf/1904.11694.pdf)
[[Project Page]](https://sites.google.com/view/neural-logic-machines)

The `jnp.einsum` op provides a DSL-based unified interface to matmul and
tensordot ops.
This `einshape` library is designed to offer a similar DSL-based approach
to unifying reshape, squeeze, expand_dims, and transpose operations.

Some examples:

* `einshape("n->n111", x)` is equivalent to `expand_dims(x, axis=1)` three times
* `einshape("a1b11->ab", x)` is equivalent to `squeeze(x, axis=[1,3,4])`
* `einshape("nhwc->nchw", x)` is equivalent to `transpose(x, perm=[0,3,1,2])`
* `einshape("mnhwc->(mn)hwc", x)` is equivalent to a reshape combining
  the two leading dimensions

## Usage

Jax version:

```py
from einshape import jax_einshape as einshape
from jax import numpy as jnp

a = jnp.array([[1, 2], [3, 4]])
b = einshape("ij->(ij)", a)
# b is [1, 2, 3, 4]
```