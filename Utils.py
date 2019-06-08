import numpy as np


def read_fashion_mnist_train_data(test_percentag=0.15):
    data = np.loadtxt("train_x", dtype=float) / 255
    labels = np.loadtxt("train_y",dtype=int)
    n = len(labels)
    n_test = int(test_percentag * n)
    perm = np.random.permutation(n)
    train_data = data[n_test:]
    train_labels = labels[n_test:]
    test_data = data[:n_test]
    test_labels = labels[:n_test]

    return train_data, train_labels, test_data, test_labels
