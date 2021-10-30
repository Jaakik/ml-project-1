"""
For running scripts
"""
from data_utils import *
from cost_functions import *
from train_utils import *
import numpy as np

data_path = "/Users/marouanejaakik/Desktop/courses/ml-project-1/data/train.csv"

# Raw data
y, x, ids = load_csv_data(data_path)

print(x.shape)

# list of lambdas put your values here
# example
lambdas = [0.001, 0.05]
degrees = [2, 3, 5]

best_degree, best_lamb, accu = train_grid_search(y, x, ridge_regression, lambdas,degrees, k_fold=10)

print(best_degree)
