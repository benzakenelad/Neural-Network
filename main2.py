import Utils
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import NerualNetwork


def main():
    X, y, test_data, test_labels = Utils.read_fashion_mnist_train_data()
    # data = load_iris()
    # X, y = data.data, data.target
    X = StandardScaler().fit_transform(X)
    nn = NerualNetwork.NN([784, 256, 10])
    nn.fit(X, y, batch_size=32, epochs=100000)


if __name__ == '__main__':
    main()
