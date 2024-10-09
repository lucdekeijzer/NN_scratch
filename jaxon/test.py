#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:39:55 2024

Copyright (c) 2024, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import jax
import jax.test_util
import jax.numpy as jnp

__all__ = ["model_backward"]


eps = 5e-2
div_lim = 5e-8


def _rel_error(X, Xhat, eps=5e-8):
    """Compute relative error, if possible, otherwise the absolute error."""
    norm_X = jnp.linalg.norm(X)
    if norm_X > eps:
        rel_error = jnp.linalg.norm(X - Xhat) / norm_X
    else:
        rel_error = jnp.linalg.norm(X - Xhat)

    return rel_error


def _loss_from_layer(model, layer_name, *inputs, params=None):
    """Compute the loss of a Model starting from a given layer name.

    The inputs must be the the layer input(s) (A) and target output (Y).
    """
    A, Y = inputs

    layer_i = next(i
                   for i, layer in enumerate(model.hypothesis.layers)
                   if layer[0] == layer_name)

    x = A
    for i in range(layer_i, len(model.hypothesis.layers)):
        _, layer = model.hypothesis.layers[i]
        if isinstance(x, (tuple, list)):
            if i == layer_i:
                x = layer(*x, params=params)
            else:
                x = layer(*x)
        else:
            if i == layer_i:
                x = layer(x, params=params)
            else:
                x = layer(x)

    loss = model.loss(Y, x)

    return loss


def model_backward(model, X, Y):
    """Test whether the model's backwards function is implemented correctly.

    Parameters
    ----------
    model : jaxon.models.Model
        The model to test.
    X : jax.Array
        The input data.
    Y : jax.Array
        The output data.

    Returns
    -------
    Nothing.

    Throws
    ------
    AssertionError
        If the provided partial derivatives are wrong.
    """
    grad = model.backward(X, Y)

    for layer_name in [layer_name
                       for layer_name, _ in model.hypothesis.layers] \
            + ["loss"]:

        # print(layer_name)

        if layer_name == "loss":
            layer_prev = model.hypothesis.layers[-1]
            output_prev = model.hypothesis.output[layer_prev[0]]

            for argnum in range(2):
                D = jax.jacrev(model.loss, argnum)(Y, output_prev)
                Dhat = grad["loss"]["inputs"][argnum]

                rel_err = _rel_error(D, Dhat, eps=div_lim)

                assert rel_err < eps
        else:
            # Find the layer
            layer_prev = None
            layer_curr = None
            for i, layer in enumerate(model.hypothesis.layers):
                if layer[0] == layer_name:
                    layer_curr = model.hypothesis.layers[i]
                    if i - 1 >= 0:
                        layer_prev = model.hypothesis.layers[i - 1]
                    break

            # Partial derivatives of the loss with respect to layer inputs
            if layer_prev is None:
                output_prev = X  # Then the input data are the inputs to layer
            else:
                output_prev = model.hypothesis.output[layer_prev[0]]

            def loss(A):
                return _loss_from_layer(model, layer_curr[0], A, Y)

            D = jax.jacrev(loss)(output_prev)
            Dhat = grad[layer_curr[0]]["inputs"][0]

            rel_err = _rel_error(D, Dhat, eps=div_lim)

            assert rel_err < eps

            # Partial derivatives of the loss with respect to layer parameters
            params = layer_curr[1].params
            if len(params) > 0:

                if layer_prev is None:
                    output_prev = X  # Then input data are the inputs to layer
                else:
                    output_prev = model.hypothesis.output[layer_prev[0]]

                def forward_dict(params, inputs):
                    return _loss_from_layer(model,
                                            layer_curr[0],
                                            *inputs,
                                            params=params)

                D_all = jax.jacrev(forward_dict)(params, (output_prev, Y))
                # Check the partial derivatives with respect to each weight
                for key in D_all:
                    Dhat = grad[layer_curr[0]][key]
                    D = D_all[key]

                    rel_err = _rel_error(D, Dhat, eps=div_lim)

                    assert rel_err < eps
