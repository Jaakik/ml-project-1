"""
For running scripts
"""
from data_utils import *
from cost_functions import *
import numpy as np

data_path = "/Users/marouanejaakik/Desktop/courses/ml-project-1/data/train.csv"

# Raw data
y , tx, ids = load_csv_data(data_path)


r_pca = PCA(tx,0.95)

print(r_pca.shape)

# intial weights
#wi = np.ones(tx.shape[1])

#local_w, loss = logistic_regression(y, tx, wi.copy(), 5, 0.01)

#print(loss)







