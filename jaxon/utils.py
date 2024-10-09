#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:06:06 2024

Copyright (c) 2024, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import functools

import jax
import jax.numpy as jnp

dtype = jnp.float32
eps = jnp.finfo(dtype).eps

__all__ = ["dtype", "eps",
           "prng_key", "check_prng_key",
           "handle_init", "handle_forward", "handle_backward",
           ]


def prng_key(seed):
    """Create a PRNG key from a given seed."""
    if hasattr(jax.random, "key"):
        return jax.random.key(seed)
    else:
        return jax.random.PRNGKey(seed)


def check_prng_key(key):
    """Try to make sense of a provided PRNG key."""
    if key is None:
        key = prng_key(0)
    elif isinstance(key, int):
        key = prng_key(key)
    elif (hasattr(key, "dtype")
            and jax.dtypes.issubdtype(key.dtype, jnp.uint32)
            and key.shape == (2,)):
        key = key
    elif (hasattr(key, "dtype") and hasattr(jax.dtypes, "prng_key")
            and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)):
        key = key
    elif (hasattr(jax, "_src") and hasattr(jax._src, "prng")
            and isinstance(key, jax._src.prng.PRNGKeyArrayImpl)):
        key = key
    else:
        try:
            key = prng_key(key)
        except TypeError:
            raise RuntimeError(f"Unknown PRNG key provided ({key}).")

    return key


def handle_init(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]

        if not self._is_initialised:

            return_values = func(*args, **kwargs)

            if not self._is_initialised:
                self._is_initialised = True

            return return_values

    return wrapper


def handle_forward(_func=None, *, num_inputs=None):
    def decorator_handle_forward(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            inputs = args[1:]

            if isinstance(num_inputs, int):
                if len(inputs) != num_inputs:
                    raise RuntimeError(f"In forward(...): Expected "
                                       f"{num_inputs} inputs, but got "
                                       f"{len(inputs)}!")
            import jaxon.models
            if isinstance(self, jaxon.models.Model):
                if len(self.hypothesis) == 0:
                    raise RuntimeError("In forward(...): Model doesn't have "
                                       "any layers.")

            self.init(*inputs)

            # If not specified already, fill in the params argument in forward
            if ("params" not in kwargs) or (kwargs["params"] is None):
                kwargs["params"] = self.params

            # Run the forward function
            return_values = func(*args, **kwargs)

            if not self._has_run_forward:
                self._has_run_forward = True

            return return_values

        return wrapper

    if _func is None:
        return decorator_handle_forward
    else:
        return decorator_handle_forward(_func)


def handle_backward(_func=None, *, num_inputs=None):
    def decorator_handle_backward(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            inputs = args[1:]

            if isinstance(num_inputs, int):
                if len(inputs) != num_inputs:
                    raise RuntimeError(f"In forward(...): Expected "
                                       f"{num_inputs} inputs, but got "
                                       f"{len(inputs)}!")

            if not self._has_run_forward:
                raise RuntimeError("In backward(...): You must run `forward` "
                                   "first!")

            # Run the backward function
            return_values = func(*args, **kwargs)

            return return_values

        return wrapper

    if _func is None:
        return decorator_handle_backward
    else:
        return decorator_handle_backward(_func)
