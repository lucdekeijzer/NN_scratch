#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:10:41 2024

Copyright (c) 2024, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc

import jax
import jax.numpy as jnp

import jaxon.utils

__all__ = ["Layer",
           "Dense",
           "ReLU",
           "Softmax",
           ]


class Layer(metaclass=abc.ABCMeta):
    """Base class for layers."""

    def __init__(self, jacrev=True, key=None):
        self._jacrev = bool(jacrev)
        self._has_run_forward = False
        self._is_initialised = False

        self.params = dict()
        self.key = jaxon.utils.check_prng_key(key)

    @jaxon.utils.handle_init
    def init(self, *inputs):
        # Note: Layer parameters can not be called "inputs" nor "loss", these
        #       names are reserved for the actual inputs to the
        #       forward/backward methods and to loss functions in Models.
        pass

    @abc.abstractmethod
    def forward(self, *inputs, params=None):
        pass

    @jaxon.utils.handle_backward
    def backward(self, *inputs):
        # TODO: How do we do to make a clever choice here?
        if self._jacrev:
            jac = jax.jacrev
        else:
            jac = jax.jacfwd

        # Compute jacobians with respect to layer paramters (if any)
        if len(self.params) > 0:
            def forward_dict(params, inputs):
                return self.forward(*inputs, params=params)

            jacobians = jac(forward_dict)(self.params, inputs)

            if "inputs" in jacobians:
                raise ValueError("In backward(...): Name reserved for inputs. "
                                 "Weights can not be called `inputs`.")
            if "loss" in jacobians:
                raise ValueError("In backward(...): Name reserved for loss. "
                                 "Weights can not be called `loss`.")
        else:
            jacobians = dict()

        _index = [None]

        # Compute jacobian with respect to the layer's inputs
        def forward_inputs(X):
            _inputs = list(inputs[:])
            _inputs[_index[0]] = X
            return self.forward(*_inputs, params=self.params)

        _jacobians = []
        for i, input_ in enumerate(inputs):
            _index[0] = i
            jacobian = jac(forward_inputs)(input_)
            _jacobians.append(jacobian)
        jacobians["inputs"] = _jacobians

        return jacobians

    def __call__(self, *inputs, params=None):
        return self.forward(*inputs, params=params)


class Dense(Layer):
    """A dense (fully connected) layer."""

    def __init__(self, d_out, jacrev=True, key=None):
        super().__init__(jacrev=jacrev, key=key)

        self.d_out = max(1, int(d_out))
        self.d_in = None

    @jaxon.utils.handle_init
    def init(self, *inputs):
        X = inputs[0]

        if len(X.shape) != 2:
            raise RuntimeError("In Dense.init(...): The input array must be "
                               "two-dimensional.")

        self.d_in = X.shape[1]  # Number of input features

        self.params = {}
        self.key, subkey = jax.random.split(self.key)
        self.params["W"] = jax.random.normal(subkey, (self.d_in, self.d_out))
        self.key, subkey = jax.random.split(self.key)
        self.params["b"] = jax.random.normal(subkey, (1, self.d_out))

    @jaxon.utils.handle_forward(num_inputs=1)
    # @functools.partial(jax.jit, static_argnums=0)
    def forward(self, *inputs, params=None):
        X = inputs[0]
        return jnp.dot(X, params["W"]) + params["b"]


class ReLU(Layer):
    """The rectified linear unit activation function."""

    def __init__(self, jacrev=True):
        super().__init__(jacrev=jacrev)

    @jaxon.utils.handle_forward(num_inputs=1)
    # @functools.partial(jax.jit, static_argnums=0)
    def forward(self, *inputs, params=None):
        return jnp.maximum(0, inputs[0])


class Softmax(Layer):
    """The softmax activation function."""

    def __init__(self, axis=-1, jacrev=True):
        super().__init__(jacrev=jacrev)

        self.axis = axis

    @jaxon.utils.handle_forward(num_inputs=1)
    # @functools.partial(jax.jit, static_argnums=0)
    def forward(self, *inputs, params=None):
        # This implementation of softmax is numerically stable (ish).
        max_ = jnp.max(inputs[0], self.axis, keepdims=True)
        numerator = jnp.exp(inputs[0] - max_)
        denominator = jnp.sum(numerator, self.axis, keepdims=True)
        result = numerator / denominator

        return result
