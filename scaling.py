import numpy as np
import pandas as pd
from sklearn import preprocessing


def min_max_train_test_scaling(X_train, X_test):
    # Min Max scaling approach
    alpha = 1.0

    X_train_scale = X_train.copy()
    X_test_scale = X_test.copy()

    for col in range(X_train.shape[1]):
        col_values = X_train[:, col]

        min_xi = np.min(col_values)
        max_xi = np.max(col_values)

        # print(min_xi, max_xi)

        # Scaling on training data
        scaled_values_train = alpha*((col_values - min_xi) / (max_xi - min_xi))
        X_train_scale[:, col] = scaled_values_train

        # Scaling on test data using mean, std of train data
        scaled_values_test = alpha * \
            ((X_test[:, col] - min_xi) / (max_xi - min_xi))
        X_test_scale[:, col] = scaled_values_test

    return X_train_scale, X_test_scale


def standard_train_test_scaling(X_train, X_test, col_indexes_to_scale):
    # Standard scaling approach
    alpha = 1.0

    X_train_scale = X_train.copy()
    X_test_scale = X_test.copy()

    for col in range(X_train.shape[1]):
        if col in col_indexes_to_scale:
            col_values = X_train[:, col]

            mean = np.mean(col_values)
            standard_deviation = np.std(col_values)

            # Scaling on training data
            scaled_values_train = alpha * \
                ((col_values - mean) / standard_deviation)
            X_train_scale[:, col] = scaled_values_train

            # Scaling on test data using mean, std of train data
            scaled_values_test = alpha * \
                ((X_test[:, col] - mean) / standard_deviation)
            X_test_scale[:, col] = scaled_values_test

    return X_train_scale, X_test_scale


def sklearn_standard_scaling(X_train, X_test):
    scaler = preprocessing.StandardScaler()

    scaler.fit(X_train)

    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)

    return X_train_scale, X_test_scale
