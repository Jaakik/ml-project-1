import numpy as np
import numpy.linalg as la
from train_utils import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for linear MSE.

    Parameters
    ----------
    y : np.array
        vector representing the output variable 
    tx : np.array
        Matrix representing the input variables 
    initial_w : np.array
        initial weight vector
    max_iters : int
        number of steps to run
    gamma : float
        step size

    Returns
    -------
    (w, loss)
        Last weight vector and corresponding loss

    """
    w = initial_w
    N = len(y) # number of datapoints

    for n_iter in range(max_iters):

        e = y - np.dot(tx,w) # error vector for a given w
        gradient_vector = -np.dot(tx.T,e)/N # gradient of the cost function (MSE) at a given w

        # update w by gradient vector
        w = w-gamma*gradient_vector


    loss = compute_MSE_linreg(y, tx, w)

    return w, loss



def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient descent algorithm for linear MSE.

    Parameters
    ----------
    y : np.array
        vector representing the output variable 
    tx : np.array
        Matrix representing the input variables 
    initial_w : np.array
        initial weight vector
    max_iters : int
        number of steps to run
    gamma : float
        step size

    Returns
    -------
    (w, loss)
        Last weight vector and corresponding loss

    """
    w = initial_w
    N = len(y) # number of datapoints

    for n_iter in range(max_iters):

        e = y - np.dot(tx,w) # error vector for a given w
        random_index = np.random.randint(0, N, size=None, dtype=int) # Return a random integer from the “discrete uniform” distribution
        gradient_vector = -e[random_index]*tx[random_index]

        # update w by gradient
        w = w-gamma*gradient_vector

    loss = compute_MSE_linreg(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using SGD.
    Parameters
    """
    w = initial_w
    w = w[:,np.newaxis]
    y = y[:,np.newaxis]
    losses = []

    # Data shuffle
    data_size = y.shape[0]
    np.random.seed(2019)
    shuffled_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffled_indices]
    shuffled_tx = tx[shuffled_indices]
    shuffled_y = shuffled_y[:,np.newaxis]

    for n_iter, by, btx in zip(range(max_iters), shuffled_y, shuffled_tx):
        btx = btx[:,np.newaxis].T
        loss, gradient = logistic_regression_step(by, btx, w)
        w -= gamma * gradient
        losses.append(loss)

    return np.squeeze(w), compute_loss_logistic(y, tx, w)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using SGD.
    Parameters

    """
    w = initial_w
    w = w[:,np.newaxis]
    y = y[:,np.newaxis]
    losses = []

    # Data shuffle
    data_size = y.shape[0]
    np.random.seed(2019)
    shuffled_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffled_indices]
    shuffled_tx = tx[shuffled_indices]
    shuffled_y = shuffled_y[:,np.newaxis]

    for n_iter, by, btx in zip(range(max_iters), shuffled_y, shuffled_tx):
        btx = btx[:,np.newaxis].T
        loss, gradient = logistic_regression_step(y, tx, w)
        loss += lambda_ * np.squeeze(w.T @ w)
        gradient += 2 * lambda_ * w
        w -= gamma * gradient

    return np.squeeze(w), compute_loss_logistic(y, tx, w)
