""" This module contains the wrappers for the SCIPY optimization algorithms.
"""

# standard library
import numpy as np


def scipy_wrapper_gradient(x, critFunc):
    """ Wrapper for the gradient calculation.
    """
    # Antibugging.
    assert (isinstance(x, np.ndarray))
    assert (np.all(np.isfinite(x)))
    assert (x.dtype == 'float')
    assert (x.ndim == 1)

    # Evaluate gradient.
    grad = critFunc.evaluate(x, 'gradient')

    # Check quality.
    assert (isinstance(grad, np.ndarray))
    assert (np.all(np.isfinite(grad)))
    assert (grad.dtype == 'float')

    return grad

def scipy_wrapper_function(x, critFunc):
    """ Wrapper for most SCIPY maximization algorithms.
    """
    # Antibugging
    assert (isinstance(x, np.ndarray))
    assert (np.all(np.isfinite(x)))
    assert (x.dtype == 'float')
    assert (x.ndim == 1)

    # Evaluate likelihood
    likl = critFunc.evaluate(x, 'function')

    # Quality checks
    assert (isinstance(likl, float))
    assert (np.isfinite(likl))

    # Finishing
    return likl