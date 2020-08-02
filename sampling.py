import random
import numpy as np


def get_uniformly_sampled_data(data, t):
    row_count = data.shape[0]
    # Uniform sampling without replacement
    random_indices = np.random.choice(row_count, t, replace=False)
    data_t = data[random_indices, :]

    return data_t


def uniform_sampling(X_train, y_train, t):
    # print(X_train.shape, y_train.shape, X_train[:3], "\n\n", y_train[:3])
    train = np.column_stack((X_train, y_train))
    # print("\n\n", train.shape, train[:3])

    # Uniform sampling
    train_t = get_uniformly_sampled_data(train, t)
    # print("\n\nSampled: ", train_t.shape, train_t[:5])
    return train_t[:, 0:-1], train_t[:, -1]


def get_oversampled_data(train_rpl, train_label, rows_per_label, row_count):
    left_rows = rows_per_label - row_count

    while (left_rows > 0):
        if left_rows <= row_count:
            train_os = get_uniformly_sampled_data(train_label, left_rows)
        else:
            train_os = get_uniformly_sampled_data(train_label, row_count)

        # train_rpl = np.append(train_rpl, train_os, axis=0)
        train_rpl = np.row_stack((train_rpl, train_os))

        left_rows -= row_count

    # print("Final sampled shape: ", train_rpl.shape)

    return train_rpl


def balanced_uniform_sampling(X_train, y_train, t):
    # print(X_train[:3], "\n\n", y_train[:3])
    train = np.column_stack((X_train, y_train))
    # print("\n\n", train[:3])

    # Balanced uniform sampling
    unique_labels = set(y_train.tolist())
    rows_per_label = int(t/len(unique_labels))
    # print("Rows per label: ", rows_per_label)
    i = 0

    for label in unique_labels:
        train_label = train[train[:, -1] == label]
        row_count = train_label.shape[0]
        # print("label: ", label, "\trows found: ", row_count)
        if rows_per_label <= row_count:
            # Sample uniformly
            train_rpl = get_uniformly_sampled_data(train_label, rows_per_label)
        else:
            # Sample uniformly and oversample
            train_rpl = get_uniformly_sampled_data(train_label, row_count)
            train_rpl = get_oversampled_data(
                train_rpl, train_label, rows_per_label, row_count)

        if i:
            train_t = np.row_stack((train_t, train_rpl))
        else:
            train_t = train_rpl.copy()

        i += 1

    # print("Balanced training data: ", train_t.shape, "\n", train_t)

    return train_t[:, 0:-1], train_t[:, -1]
