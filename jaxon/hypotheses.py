#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:50:39 2024

Copyright (c) 2024, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc

import jax

import jaxon.utils
import jaxon.layers

__all__ = ["Hypothesis",
           "FeedForwardNeuralNetwork",
           "AffineFunction",
           ]


class Hypothesis(jaxon.layers.Layer):
    def __init__(self, jacrev=True, key=None):
        super().__init__(jacrev=jacrev, key=key)

        self.layers = []
        self.jacobians = {}

    def add(self, name, layer):
        try:
            self.get(name)
            raise ValueError(f"In NeuralNetwork.add(...): Layer name already "
                             f"taken: '{name}'")
        except ValueError:
            # If not existing already, add it
            self.layers.append((name, layer))

    def get(self, name):
        for layer_name, layer in self.layers:
            if name == layer_name:
                return layer

        raise ValueError(f"In NeuralNetwork.get(...): Unknown layer name: "
                         f"'{name}'")

    def delete(self, name):
        found = None
        for i, (layer_name, layer) in enumerate(self.layers):
            if name == layer_name:
                found = i
                break
        if found is not None:
            value = self.layers[found]
            del self.layers[found]
            return value

        raise ValueError(f"In NeuralNetwork.delete(...): Unknown layer name: "
                         f"'{name}'")

    def __len__(self):
        return len(self.layers)

    @abc.abstractmethod
    def forward(self, *inputs, params=None):
        pass

    @abc.abstractmethod
    def backward(self, *inputs):
        pass


class FeedForwardNeuralNetwork(Hypothesis):
    def __init__(self, jacrev=True, key=None):
        super().__init__(jacrev=jacrev, key=key)

    @jaxon.utils.handle_forward(num_inputs=1)
    def forward(self, *inputs, params=None):
        # super().forward(*inputs, params=None)

        x = inputs

        self.output = {}
        for name, layer in self.layers:
            if isinstance(x, (tuple, list)):
                x = layer(*x)
            else:
                x = layer(x)
            self.output[name] = x

        output_name, _ = self.layers[-1]

        return self.output[output_name]

    @jaxon.utils.handle_backward(num_inputs=1)
    def backward(self, *inputs):

        for L in list(range(len(self.layers) - 1, 0, -1)):
            name_curr, layer_curr = self.layers[L]
            name_prev, layer_prev = self.layers[L - 1]
            J = layer_curr.backward(self.output[name_prev])
            self.jacobians[name_curr] = J

        # Do it also for the last layer wrt. the inputs
        name_curr, layer_curr = self.layers[0]
        J = layer_curr.backward(*inputs)
        self.jacobians[name_curr] = J

        return self.jacobians


class AffineFunction(FeedForwardNeuralNetwork):
    """An hypothesis set of affine functions."""

    def __init__(self, jacrev=True, key=None):
        super().__init__(jacrev=jacrev, key=key)

        self.key, subkey = jax.random.split(self.key)
        self.add("dense", jaxon.layers.Dense(1, key=subkey))
