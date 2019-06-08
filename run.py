from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import NerualNetwork

np.random.seed(0)


def main():
    # data loading
    data = load_digits()
    X, y = data.data, data.target
    n = len(y)
    train_amount = int(0.85 * n)

    # normalization
    X = StandardScaler().fit_transform(X)

    # data shuffle
    perm = np.random.permutation(n)
    X, y = X[perm], y[perm]

    # define train and test
    train_x, test_x = X[:train_amount], X[train_amount:]
    train_y, test_y = y[:train_amount], y[train_amount:]

    # training
    nn = NerualNetwork.NN(network_architecture=[64, 128, 10])  # 10 classes output, 64 dim data
    nn.fit(train_x, train_y, batch_size=16, epochs=128, print_evaluation=True)

    # evaluation on test
    predictions = nn.predict(test_x)
    test_acc = accuracy_score(test_y, predictions)
    print(f'test accuracy : {test_acc}')


if __name__ == '__main__':
    main()
