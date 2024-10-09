#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:07:12 2024

Copyright (c) 2024, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import itertools
import jaxon

SEED = 42

PLOT_SETTINGS = {"text.usetex": True,
                 "font.family": "serif",
                 "figure.figsize": (10.0, 5.0),
                 "font.size": 16,
                 "axes.labelsize": 16,
                 "legend.fontsize": 14,
                 "xtick.labelsize": 14,
                 "ytick.labelsize": 14,
                 "axes.titlesize": 24,
                 "lines.linewidth": 2.0,
                 "axes.formatter.limits": [-5, 5],
                 }
plt.rcParams.update(PLOT_SETTINGS)


# Example 1: A linear regression model.

data = jnp.load("regression_data.npz")
X = data["X"]
y = data["y"]

n, p = X.shape
c = y.shape[1]

key = jaxon.utils.prng_key(SEED)  # Generate PRNG key
key, subkey = jax.random.split(key)  # Use the split template for new rng keys
affine_hypothesis = jaxon.hypotheses.AffineFunction(key=subkey)
loss = jaxon.losses.MeanSquaredError()


class LinearRegressionModel(jaxon.models.Model):
    print("-------------------LinearRegressionModel class is being used")
    """A linear regression model."""

    @jaxon.utils.handle_backward(num_inputs=2)
    def backward(self, *inputs):

        """Implement the backwards pass, compute all derivatives."""
        X, y = inputs  # The training data

        # Get all layers' Jacobians
        J = self.hypothesis.backward(X)

        # Store the gradients in a dictionary
        grad = {}

        # Compute the gradient of the loss function, wrt. both its arguments
        grad["loss"] = self.loss.backward(y, self.output["hypothesis"])

        # Here you need to implement the backwards steps for all layers. Note
        # that you need to compute derivatives for both layer weights _and_
        # layer inputs.

        # ""W = J['W']
        # b = J['b']
        # inputs = J['inputs'][0]

        # d_loss_d_output = grad["loss"]
        
        # derivatives = {}

        # # Derivates calculation for weights
        # inputs_reshaped = inputs.reshape(inputs.shape[0], -1)
        # d_loss_d_output_reshaped = d_loss_d_output.reshape(d_loss_d_output.shape[0], -1)
        # dL_dW = np.dot(d_loss_d_output_reshaped.T, inputs_reshaped)
        # derivatives['W'] = dL_dW.reshape(W.shape)""

        def compute_partial_derivatives(layer_dict):
            print("we got to function 2")
            partials = {}
            
            W = layer_dict['W']
            inputs = layer_dict['inputs'][0]  # Assuming there's only one input array
            
            print("we got here")
            # Compute partial derivative for weights (∂y/∂W)
            # This is equivalent to the input for each output unit
            dY_dW = jnp.einsum('ijkl->ijkl', inputs)
            partials['W'] = dY_dW
            print(f"partials in func 2{partials=}")

            
            # Compute partial derivative for inputs (∂y/∂X)
            # This is equivalent to the weights for each output unit
            print("after partials")
            dY_dX = jnp.einsum('lm->ijkm', W)
            print("after dydx")
            partials['inputs'] = [dY_dX]  # Put in a list as required
            print(f"partials in func 2{partials=}")
            return partials

        def structure_partial_derivatives(layers):
            grad = {}
            print("do we get here??!?!?!?!")
            # Compute and add partial derivatives for each layer
            for layer_name, layer_dict in layers.items():
                print(layer_name)
                # print(layer_dict)
                layer_partials = compute_partial_derivatives(layer_dict)
                print(type(layer_partials))
                grad[layer_name] = {
                    "W": layer_partials["W"],
                    "inputs": layer_partials["inputs"]
                }
            print("We got here yaaaayyy")
            return grad
        grad = structure_partial_derivatives(J)


        return grad


model = LinearRegressionModel(loss, affine_hypothesis)
yhat = affine_hypothesis(X)
loss_value = model(X, y)

# Test the computed derivatives
try:
    jaxon.test.model_backward(model, X, y)
    print(f"Great! The derivatives provided by '{type(model).__name__}' "
          "seem correct!")
except Exception:
    print(f"Oops! The derivatives provided by '{type(model).__name__}' seem "
          "to be wrong!")


# Here you need to implement gradient descent using your partial derivatives

# Plot the training loss
fig = plt.figure()
fig.subplots_adjust(left=0.0575,
                    right=0.985,
                    top=0.93,
                    bottom=0.1075,
                    wspace=0.00,
                    hspace=0.12)
plt.plot(range(1, len(training_loss) + 1), training_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.title("Gradient Descent on a Linear Regression Model")
plt.xlim([1, 200])


# Example 2: A deep neural network model for classification.

data = jnp.load("classification_data.npz")
X = data["X"]
Y = data["y"]  # Note: Captial Y here!

n, p = X.shape
c = Y.shape[1]

key = jaxon.utils.prng_key(SEED)  # Generate PRNG key

key, subkey = jax.random.split(key)
nn = jaxon.hypotheses.FeedForwardNeuralNetwork(key=subkey)

# Implement your neural network here using nn.add(...)


class NeuralNetworkModel(jaxon.models.Model):
    """A neural network model for multi-class classification."""

    @jaxon.utils.handle_backward(num_inputs=2)
    def backward(self, *inputs):
        """Implement the backwards pass, compute all derivatives."""
        X, Y = inputs  # The training data

        # Get all layers' Jacobians
        J = self.hypothesis.backward(X)

        # Store the gradients in a dictionary
        grad = {}

        # Compute the gradient of the loss function, wrt. both its arguments
        grad["loss"] = self.loss.backward(Y, self.output["hypothesis"])

        # Here you need to implement the backwards steps for all layers. Note
        # that you need to compute derivatives for both layer weights _and_
        # layer inputs.

        return grad


model = NeuralNetworkModel(loss, nn)
Yhat = nn(X)
loss_value = model(X, Y)

# Test the computed derivatives
try:
    jaxon.test.model_backward(model, X, Y)
    print(f"Great! The derivatives provided by '{type(model).__name__}' "
          "seem correct!")
except Exception:
    print(f"Oops! The derivatives provided by '{type(model).__name__}' seem "
          "to be wrong!")


# Here you need to implement gradient descent using your partial derivatives

# Plot the training loss
fig = plt.figure()
fig.subplots_adjust(left=0.0575,
                    right=0.985,
                    top=0.93,
                    bottom=0.1075,
                    wspace=0.00,
                    hspace=0.12)
plt.plot(range(1, len(training_loss) + 1), training_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.title("Gradient Descent on a Neural Network for Multiclass Classification")
plt.xlim([1, 200])
