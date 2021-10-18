import numpy as np
import numpy.linalg as la
from train_utils import *


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


def least_squares(y, tx):
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T.dot(y))
    err = compute_loss(y,tx,w)
    return err, w

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    w = np.linalg.inv(tx.T.dot(tx) + 2 * lambda_ * tx.shape[0] * np.identity(tx.shape[1])).dot(tx.T.dot(y))
    err = compute_loss(y,tx,w)
    return err, w
