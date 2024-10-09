#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:10:14 2024

Copyright (c) 2024, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc

import jax.numpy as jnp

import jaxon.utils
import jaxon.layers

__all__ = ["Loss",
           "MeanSquaredError", "CategoricalCrossentropy",
           ]


class Loss(jaxon.layers.Layer):
    """Base class for loss functions."""

    def __init__(self, reduce=jnp.mean, jacrev=True, key=None):
        super().__init__(jacrev=jacrev, key=key)

        if not callable(reduce):
            raise ValueError("In CategoricalCrossentropy(...): Note that "
                             "`reduce` must be callable.")
        self.reduce = reduce

    @abc.abstractmethod
    def forward(self, y_true, y_pred, params=None):
        pass


class MeanSquaredError(Loss):
    """The mean squared error loss function."""

    def __init__(self, reduce=jnp.mean, jacrev=True, **kwargs):
        super().__init__(reduce=reduce, jacrev=jacrev, **kwargs)

    @jaxon.utils.handle_forward(num_inputs=2)
    # @functools.partial(jax.jit, static_argnums=0)
    def forward(self, *inputs, params=None):
        y_true, y_pred = inputs

        # Reduce over all dimensions
        # axes = list(range(0, y_true.ndim))
        result = self.reduce(jnp.square(y_true - y_pred),
                             # axis=axes,
                             keepdims=False)

        return result


class CategoricalCrossentropy(Loss):
    """The categorical cross-entropy loss function."""

    def __init__(self, reduce=jnp.mean, jacrev=True, **kwargs):
        super().__init__(reduce=reduce, jacrev=jacrev, **kwargs)

    @jaxon.utils.handle_forward(num_inputs=2)
    # @functools.partial(jax.jit, static_argnums=0)
    def forward(self, *inputs, params=None):
        y_true, y_pred = inputs

        y_pred = jnp.clip(y_pred, jaxon.utils.eps, 1.0 - jaxon.utils.eps)

        # Sum over class dimension(s)
        axes = list(range(1, y_true.ndim))
        result = -jnp.sum(y_true * jnp.log(y_pred), axis=axes, keepdims=False)

        # Reduce over mini-batch dimension
        return self.reduce(result)
