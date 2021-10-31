from data_utils import *
from implementations import *
from train_utils import *
import numpy as np

# Raw data
print("loading data...")
y, x_train, ids_train = load_csv_data("data/train.csv")
_, x_test, ids_test = load_csv_data("data/test.csv")

best_lambda_0 = 1e-09
best_lambda_1 = 1e-09
best_lambda_2 = 0.04
best_lambda_3 = 0.04
best_degree = 5
pred = np.zeros(x_test.shape[0])

# split by pri jet category and preprocessing step
print("preprocessing...")
mask_0, mask_1, mask_2, mask_3 = split_data_jet_mask(x_train)
mask_t_0, mask_t_1, mask_t_2, mask_t_3 = split_data_jet_mask(x_test)
X_0, X_t_0 = preprocess(x_train[mask_0], x_test[mask_t_0], y[mask_0], best_degree)
X_1, X_t_1 = preprocess(x_train[mask_1], x_test[mask_t_1], y[mask_1], best_degree)
X_2, X_t_2 = preprocess(x_train[mask_2], x_test[mask_t_2], y[mask_2], best_degree)
X_3, X_t_3 = preprocess(x_train[mask_3], x_test[mask_t_3], y[mask_3], best_degree)

# Best model
print("training...")
w_0, _ = ridge_regression(y[mask_0], X_0, best_lambda_0)
w_1, _ = ridge_regression(y[mask_1], X_1, best_lambda_1)
w_2, _ = ridge_regression(y[mask_2], X_2, best_lambda_2)
w_3, _ = ridge_regression(y[mask_3], X_3, best_lambda_3)

print("inference...")
pred[mask_t_0] = predict_labels(w_0, X_t_0)
pred[mask_t_1] = predict_labels(w_1, X_t_1)
pred[mask_t_2] = predict_labels(w_2, X_t_2)
pred[mask_t_3] = predict_labels(w_3, X_t_3)


create_csv_submission(ids_test, pred, "final_sub.csv")
