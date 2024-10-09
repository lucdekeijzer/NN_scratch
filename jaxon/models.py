#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:30:10 2024

Copyright (c) 2024, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc
import jax
import jaxon.utils
import jaxon.layers

__all__ = ["Model"]


class Model(jaxon.layers.Layer):
    """A base class for models (hypothesis + loss)."""

    def __init__(self, loss, hypothesis, jacrev=True, key=None):
        super().__init__(jacrev=jacrev, key=key)

        assert isinstance(loss, jaxon.layers.Layer)
        self.loss = loss
        assert isinstance(hypothesis, jaxon.layers.Layer)
        self.hypothesis = hypothesis

        self.params = _UninitialisedParams()

    def _setup_params(self):
        self.params = dict()
        self.params["loss"] = self.loss.params
        for layer_name, layer in self.hypothesis.layers:
            self.params[layer_name] = layer.params

    @jaxon.utils.handle_init
    def init(self, *inputs):
        pass

    @jaxon.utils.handle_forward(num_inputs=2)
    def forward(self, *inputs, params=None):
        X, y = inputs

        self.output = {}
        self.output["hypothesis"] = self.hypothesis(X)
        self.output["loss"] = self.loss(y, self.output["hypothesis"])

        self._setup_params()

        return self.output["loss"]

    @abc.abstractmethod
    def backward(self, *inputs):
        X, y = inputs

        # Forward pass (recompute for clarity)
        self.output["hypothesis"] = self.hypothesis(X)
        self.output["loss"] = self.loss(y, self.output["hypothesis"])

        # Initialize gradient dictionary
        grads = {}

        # Compute gradient for loss
        grads["loss"] = jax.grad(self.output["loss"], argnums=(0, 1))

        # Compute gradients for hypothesis layers
        for layer_name, layer in self.hypothesis.layers:
            grads[layer_name] = {}
            grads[layer_name]["inputs"] = jax.grad(layer.output, argnums=0)
            grads[layer_name]["params"] = jax.grad(layer.output, argnums=1)

        return grads


class _UninitialisedParams(dict):
    """Used to give a warning message if forward has not yet been run.

    If the user tries to access a model's params attribute before running
    forward, this class will be used and throw a RuntimeError.
    """

    def __init__(self):
        super().__init__()

    def __getitem__(self, key):
        raise RuntimeError("In Model.params: You must run `forward` first!")

    def __setitem__(self, key, value):
        raise RuntimeError("In Model.params: You must run `forward` first!")
