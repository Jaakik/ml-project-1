import numpy as np

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
    N = len(y) # number of datapoints
    e = y - np.dot(tx,w) # error vector
    MSE = np.dot(e.T,e)/(2*N)
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
    return 1/2*np.mean(e**2)


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
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred

def accuracy(features, w, true_y):
    """
    accuracy: calculates the accuracy of a prediction
    @input:
    - np.array(N,m) features
    - np.array(m,) weights
    - np.array(N,) true_y
    @output: (TP+TN)/Total
    """
    y_pred = predict_labels(w, features)
    #encode to 0/1
    y_pred_enc = (y_pred + 1) / 2
    P_N = len(y_pred_enc[np.where(np.subtract(y_pred_enc, true_y) == 0)])
    return (P_N / len(true_y)) * 100


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

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def train_model(X, y, initial_w, K , MAX_ITERS, verbose=True):
    """
    to be implemented for training and cross validating
    """





