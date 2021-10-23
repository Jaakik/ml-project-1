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


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


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


def replace_w_mean(X,y):
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



