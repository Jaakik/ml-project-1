# -*- coding: utf-8 -*-
"""some functions to load the data and submit on aicrowd taken from the course labs ."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def standardize(x):
    """standardize data
    @input:
    - np.array(N,m) x: features
    @output: np.array(N,m) with standardized data
    """
    # if bias column:
    if np.array_equal(x[:, 0], np.ones(len(x))):
        centered_data = x[:, 1:] - np.mean(x[:, 1:], axis=0)
        std_data = centered_data / np.std(centered_data, axis=0)
        return np.hstack((np.ones((len(x), 1)), std_data))
    else:
        centered_data = x - np.mean(x, axis=0)
        std_data = centered_data / np.std(centered_data, axis=0)
        return std_data


def min_max_normalize(x):
    """min max normalize
       @input:
       - np.array(N,m) x: features
       @output: np.array(N,m) with normalized data
       """
    min_f = np.amin(x, axis=0)
    max_f = np.amax(x, axis=0)
    n_features = x.shape[1]

    for i in range(n_features):
        x[:i] = (x[:i] - min_f[i]) / max_f[i] - min_f[i]


def scale(x):
    """scaler to unit vector
           @input:
           - np.array(N,m) x: features
           @output: np.array(N,m) with zero scaled data
           """
    norm = np.linalg.norm(x, axis=0)
    n_features = x.shape[1]

    for i in range(n_features):
        x[:i] = x[:i] / norm[i]


def rem_feat_by_missing_ratio(X, threshold):
    """remove data at threshold of missing ratio
               @input:
               - np.array(N,m) x: features
               - threshold : the ratio of missing value to disregard the feature
               @output: np.array(N,m') Data with removed features
               """
    remove_idx = []
    for feature in range(X.shape[1]):
        values = X[:, feature]
        missing_ratio = 1 - np.sum(values > -999) / len(values)
        if missing_ratio > threshold:
            remove_idx.append(feature)

    x = np.delete(X, remove_idx, axis=0)
    return x


def replace_w_mean(X, y):
    """replaces the missing values for every feature with mean value
                  @input:
                  - np.array(N,m) x: features
                  - np.array(N) y : labels
                  @output: np.array(N,m) Data with mean value for missing values
                  """

    for feature in range(X.shape[1]):
        mask = np.full(len(y), False, dtype=bool) | (X[:, feature] > -999)
        mean_boson = np.mean(X[np.logical_and(y == -1, mask), feature])
        mean_other = np.mean(X[np.logical_and(y == 1, mask), feature])
        X[np.logical_and(y == -1, ~mask), feature] = mean_boson
        X[np.logical_and(y == 1, ~mask), feature] = mean_other


def PCA(x, threshold):
    """applies PCA feature reduction
                     @input:
                     - np.array(N,m) x: features
                     - float (0,1) threshold for minimum percentage of data to be explained
                        the selected principal components
                     @output: np.array(N,m') data after PCA
                     """
    standardize(x)
    covariance_matrix = np.cov(x.T)

    # the principal components
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    # Calculating the explained variance on each of components
    variance_explained = []
    for i in eigen_values:
        variance_explained.append((i / sum(eigen_values)) * 100)

    # Identifying components that explain at least 95%
    cumulative_variance_explained = np.cumsum(variance_explained)

    cumulative_variance_explained /= 100

    n_components = 0

    while n_components < x.shape[1] and cumulative_variance_explained[n_components] < threshold:
        n_components += 1

    # Using two first components (because those explain more than threshold)
    projection_matrix = (eigen_vectors.T[:][:n_components]).T

    # Getting the product of original standardized X and the eigenvectors
    X_pca = x.dot(projection_matrix)

    return X_pca


def polynomial_expansion(X, degree):
    """ polynomial feature expansion
    @input:
    - np.array(N,m) X: features
    - double degree: degree of expansion
    @output: [1, X, X^2, X^3, etc]
    """
    # add bias term:
    if not np.array_equal(X[:, 0], np.ones(len(X))):
        X_poly = np.hstack((np.ones((len(X), 1)), X))
    if degree > 1:
        for deg in range(2, degree + 1):
            X_poly = np.c_[X_poly, np.power(X, deg)]
    return X_poly


def interaction_feature_first_order(X):
    product_features = []

    for feature1 in range(X.shape[1]):
        for feature2 in range(X.shape[1]):
            if feature1 < feature2:
                new_feat = X[:, feature1] * X[:, feature2]
                product_features.append(new_feat)
    return np.array(product_features).T


def add_bias(X):
    """
    adds bias term for every features
    ----------
    X: ndarray Feature matrix
    Returns ndarray tilda matrix of the feature matrix
    """
    return np.c_[np.ones(X.shape[0]), X]


def split_data_jet_mask(x):
    """
    Returns 3 masks corresponding to the rows of x where the feature 22 'PRI_jet_num'
    is equal to 0, 1 and  2 or 3 respectively.
    """
    return x[:, 22] == 0, x[:, 22] == 1, x[:, 22] == 2, x[:, 22] == 3


def remove_null_features(X):
    to_remove = []
    for feature in range(len(X[0])):
        values = X[:, feature]
        missing_ratio = 1 - np.sum(values > -999) / len(values)
        if (missing_ratio == 1.0):
            to_remove.append(feature)
    X = np.delete(X, obj=to_remove, axis=1)
    return X


def median_replacement(X, y):
    for feature in range(X.shape[1]):
        mask = np.full(len(y), False, dtype=bool) | (X[:, feature] > -999)
        median_boson = np.median(X[np.logical_and(y == -1, mask), feature])
        median_other = np.median(X[np.logical_and(y == 1, mask), feature])
        X[np.logical_and(y == -1, ~mask), feature] = median_boson
        X[np.logical_and(y == 1, ~mask), feature] = median_other
    return X


def preprocess(x_train , x_test , degree):
    """

        :param x_train:
        :param x_test:
        :param degree:
        :return:
        """
    #TODO put your preprocssing here @Julien
    #Look at how Yassine splits the data depending on the jet num
    # replace with median / mean
    #standardize
    # add bias term or not / Poly-expansion / first order poly expansion
    # Do the the same for x_train and y_test
    #Notice thata x_test will be the hold out fold in k-FOLD CROSS validation
    #you can add other parameters and you are free to do what you want here

    return x_train_processed , x_test_processed

