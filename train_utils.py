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