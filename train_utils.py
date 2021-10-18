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