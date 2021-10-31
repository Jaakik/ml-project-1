import numpy as np
from data_utils import *


def compute_MSE_linreg(y, tx, w):
    """Calculate the cost of a linear regression model using MSE.

    Parameters
    ----------
    y : np.array
        vector representing the output variable 
    tx : np.array
        Matrix representing the input variables 
    w : np.array
        vector representing the parameters of the linear regression model 

    Returns
    -------
    float
        MSE of the linear model

    """
    N = len(y)  # number of datapoints
    e = y - np.dot(tx, w)  # error vector
    MSE = np.dot(e.T, e) / (2 * N)
    return MSE


def compute_loss_logistic(y, tx, w):
    """
    compute the cost by negative log likelihood.
    """
    sigmoid_param = tx @ w
    # Thresholding to prevent any overflow
    sigmoid_param[sigmoid_param > 20] = 20
    sigmoid_param[sigmoid_param < -20] = -20
    sigm = sigmoid(sigmoid_param)
    loss = (y.T @ np.log(sigm)) + ((1 - y).T @ np.log(1 - sigm))
    return np.squeeze(-loss)


def compute_gradient_logistic(y, tx, w):
    """
    compute the gradient of loss.
    """
    sigmoid_param = tx @ w
    # Thresholding to prevent any overflow
    sigmoid_param[sigmoid_param > 20] = 20
    sigmoid_param[sigmoid_param < -20] = -20

    return tx.T @ (sigmoid(sigmoid_param) - y)


def logistic_regression_step(y, tx, w):
    """
    return the loss, gradient
    """
    return compute_loss_logistic(y, tx, w), compute_gradient_logistic(y, tx, w)


def sigmoid(t):
    """
    apply sigmoid function on t.
    """
    return 1.0 / (1 + np.exp(-t))


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e ** 2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)


def predict_labels(weights, data):
    """Generates class predictions given weights for all implementations except logistic_reg, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def predict_labels_logistic(weights, data):
    """Generates class predictions given weights for the logistic reg, and a test data matrix"""
    y_pred = sigmoid(np.dot(data, weights))
    y_pred[y_pred < 0.5] = -1
    y_pred[y_pred > 0.5] = 1

    return y_pred


def accuracy_w(features, w, true_y):
    """
    accuracy: calculates the accuracy of a prediction
    @input:
    - np.array(N,m) features
    - np.array(m,) weights
    - np.array(N,) true_y
    @output: (TP+TN)/Total
    """
    y_pred = predict_labels(w, features)
    # encode to 0/1
    y_pred_enc = (y_pred + 1) / 2
    P_N = len(y_pred_enc[np.where(np.subtract(y_pred_enc, true_y) == 0)])
    return (P_N / len(true_y)) * 100


def accuracy(y_pred, true_y):
    return np.sum(y_pred == true_y) / len(y_pred)


def build_k_indices(y, k_fold, seed=2):
    """build_k_indices: build k indices for k-fold
    @input:
    - np.array(N,) y: labels
    - double k_fold: number of k-folds (e.g. 5)
    - double seed: seed for random generator
    @output: k indices sets for k-fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_sets(tX, y, k_indices, i):
    """
    cross_validation_sets: separates tX and y randomly into training and validation sets.
    @input:
    - np.array(N,m) tx: features
    - np.array(N,) y: labels
    - list k_indices: indices for k-fold cross-val
    - i: the index of the iteration. (the i-th iteration to find the correct k_indices)
    @output:
    - np.array(percentage*N,m) tX_train: training features
    - np.array(percentage*N,) y_train: training labels
    - np.array((1-percentage)*N,m) tX_val: validation features
    - np.array((1-percentage)**N,) y_val: validation labels
    """
    train_indices = np.concatenate(np.delete(k_indices, i, axis=0), axis=0)
    val_indices = k_indices[i]

    # creates training and validation:
    tX_train = np.take(tX, train_indices, axis=0)
    y_train = np.take(y, train_indices, axis=0)
    tX_val = np.take(tX, val_indices, axis=0)
    y_val = np.take(y, val_indices, axis=0)

    # tests:
    size = len(train_indices) + len(val_indices)

    assert (tX_train.shape[0] + tX_val.shape[0] == size)
    assert (y_train.shape[0] + y_val.shape[0] == size)

    return tX_train, y_train, tX_val, y_val


def preprocess(x_train, x_test, y, degree):
    """

        :param y:
        :param x_train:
        :param x_test:
        :param degree:
        :return:
        """
    # Train
    x_train_processed = remove_null_features(x_train)
    x_train_processed = median_replacement(x_train_processed, y)
    x_train_processed = polynomial_expansion(x_train_processed, degree)


    # Test 
    x_test_processed = remove_null_features(x_test)
    x_test_processed = polynomial_expansion(x_test_processed, degree)

    return x_train_processed, x_test_processed


def train_model(x, y, opt, k_indices, k, degree, lamb=None, max_iters=None, gamma=None, is_log=False):
    """
    to be implemented for training and cross validating
    """

    # Initial weights
    wi = np.ones(x.shape[1])

    # Data
    mask_test = k_indices[k]
    mask_train = np.delete(k_indices, k, axis=0).ravel()

    x_train = x[mask_train, :]
    x_test = x[mask_test, :]
    y_train = y[mask_train]
    y_test = y[mask_test]

    # Call the preprocessing function HERE using the degree param
    x_train, x_test = preprocess(x_train, x_test, y, degree)

    # Least squares
    if lamb is None and max_iters is None and gamma is None:
        weights, _ = opt(y_train, x_train)
    # ridge regression
    elif lamb is not None and max_iters is None and gamma is None:
        weights, _ = opt(y_train, x_train, lamb)

    # ridge logistic regression
    elif max_iters is not None and gamma is not None and lamb is not None:
        weights, _ = opt(y_train, x_train, lamb, wi, max_iters, gamma)

    # Everything else
    elif max_iters is not None and gamma is not None and lamb is None:
        weights, _ = opt(y_train, x_train, wi, max_iters, gamma)

    # predictions
    if is_log:
        y_train_pred = predict_labels_logistic(weights, x_train)
        y_test_pred = predict_labels_logistic(weights, x_test)
    else:
        y_train_pred = predict_labels(weights, x_train)
        y_test_pred = predict_labels(weights, x_test)

    # compute accuracy for train and test data
    acc_train = accuracy(y_train_pred, y_train)
    acc_test = accuracy(y_test_pred, y_test)

    return acc_train, acc_test


def train_grid_search(y, x, opt, lambdas, degrees, gamma, max_iters, k_fold, is_ridge=False, is_log=False):
    """
    gets the best hyper parameters and best opt method
    :param max_iters:
    :param y:
    :param x:
    :param degrees:
    :param opt:
    :param is_log: whether opt is logistic regression or not
    :param lambdas:
    :param k_fold:
    :return:
    """

    k_indices = build_k_indices(y, k_fold)
    comparison = []
    # we can also iterate over a list of opt function but this is going to take a long time between runs
    # I prefer to do a grid search per function in this case that is the example implementation for ridge
    if is_ridge:
        for degree in degrees:
            for lamb in lambdas:
                accs_test = []
                for k in range(k_fold):
                    _, acc_test = train_model(x, y, opt, k_indices, k, degree, lamb, max_iters, gamma, is_log)
                    accs_test.append(acc_test)
                comparison.append([degree, lamb, np.mean(accs_test)])

    comparison = np.array(comparison)
    ind_best = np.argmax(comparison[:, 2])
    best_degree = comparison[ind_best, 0]
    best_lamb = comparison[ind_best, 1]
    accu = comparison[ind_best, 2]

    return best_degree, best_lamb, accu
