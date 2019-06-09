import numpy as np
import sys
from math import ceil


class NN():
    def __init__(self, network_architecture, weights_epsilon=0.1):
        '''
        :param network_architecture: the network architecture
        :param weights_epsilon: weights scaling param
        '''
        # model weights and biases
        self.layers = []
        self.biases = []

        self.network_depth = len(network_architecture) - 1
        self.network_architecture = network_architecture
        # initializing weights and biases by network architecture
        for i in range(self.network_depth):
            layer = self._generate_layer(network_architecture[i], network_architecture[i + 1], weights_epsilon)
            self.layers.append(layer)
            bias = self._generate_bias(network_architecture[i + 1], weights_epsilon)
            self.biases.append(bias)
        self.output_len = network_architecture[-1]

    def _forward(self, x, y=None):
        '''
        forwarding a sample in the model
        :param x: a data sample
        :param y: the sample's corresponding label
        :return: list of all outputs from each layer (after activation function)
        '''
        h = np.copy(x)
        h_list = [h]
        # forwarding the sample
        for i, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            if i + 1 != self.network_depth:
                h = self._relu(np.dot(layer.T, h) + bias)
            else:
                h = self._softmax(np.dot(layer.T, h) + bias)
            h_list.append(h)
        # loss calculation
        if y is not None:
            loss = self._loss(h_list[-1], y)
            return h_list, loss
        return h_list

    def _backpropagation(self, h_list, y):
        '''
        back propagating the error + gradients calculation
        :param h_list: list of all outputs from each layer (after activation function)
        :param y: the sample's corresponding label
        :return: list of all derivatives (by Wi and Bi)
        '''
        error_list = []
        h_outout = h_list.pop(-1)
        error = (h_outout - y) * self._softmax_dervative(h_outout)
        # back propagating the error
        for h, W in zip(reversed(h_list), reversed(self.layers)):
            error_list.append(error)
            error = np.dot(W, error) * self._relu_dervative(h)

        dLdW_list = []
        dLdB_list = []
        # derivatives calculating
        for h, delta in zip(h_list, reversed(error_list)):
            dLdW = np.dot(h.reshape(-1, 1), delta.reshape(-1, 1).T)
            dLdB = delta
            dLdW_list.append(dLdW)
            dLdB_list.append(dLdB)

        return dLdW_list, dLdB_list

    def fit(self, X, Y, epochs=1024, batch_size=32, eta=0.1):
        '''
        fitting the model to X and Y
        :param X: data
        :param Y: labels
        :param epochs: number of epochs
        :param batch_size: batch size
        :param eta: learning rate
        :param decay: weights decay (regularization)
        '''
        n = len(Y)
        Y = self._generate_onehot(Y)
        n_batch = int(n / batch_size) + 1
        for i in range(epochs):
            perm = np.random.permutation(n)
            X, Y = X[perm], Y[perm]
            for batch_idx in range(n_batch):
                # initialization
                layers_gradients, biases_gradients = self._generate_initialized_gradient_matrices()
                # batches preparation
                batch_data = X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                batch_targets = Y[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                # batch
                for x, y in zip(batch_data, batch_targets):
                    # forwarding a sample
                    outputs_list, loss = self._forward(x, y)
                    # back propagating the sample's error
                    dLdW, dLdb = self._backpropagation(outputs_list, y)
                    # updating the cumulative gradients
                    for dW, new_dW in zip(layers_gradients, dLdW):
                        dW += new_dW
                    for db, new_db in zip(layers_gradients, dLdb):
                        db += new_db
                # division by batch size
                for dW, db in zip(layers_gradients, biases_gradients):
                    dW /= batch_size
                    db /= batch_size
                # updating the weights and biases (gradient descent)
                for layer, dW in zip(self.layers, layers_gradients):
                    layer -= eta * dW
                for bias, dB in zip(self.biases, biases_gradients):
                    bias -= eta * dB
            # training data evaluation
            loss, acc = self.evaluation(X, Y)
            print(f'epoch {i} train loss : {loss}, train accuracy : {acc}')

    def _loss(self, y_hat, y_true):
        # loss function
        J = np.sum((-1.0) * (y_true * np.log(y_hat) + (1.0 - y_true) * np.log(1.0 - y_hat)))
        return J

    def _softmax(self, x):
        # softmax activation function
        return 1.0 / (1.0 + np.exp(-x))

    def _generate_layer(self, input_len, output_len, weights_epsilon):
        # generating an model's fully connected layer
        return (np.random.rand(input_len, output_len) * 2 * weights_epsilon) - weights_epsilon

    def _generate_bias(self, output_len, weights_epsilon):
        # generating an model's bias layer
        return (np.random.rand(output_len) * 2 * weights_epsilon) - weights_epsilon

    def _generate_onehot(self, labels):
        # convert y vector to one hot matrix
        n = len(labels)
        onehot_matrix = np.zeros((n, self.output_len))
        for i in range(n):
            onehot_matrix[i, labels[i]] = 1.0
        return onehot_matrix

    def _softmax_dervative(self, x):
        # softmax derive function
        return x * (1.0 - x)

    def _generate_initialized_gradient_matrices(self):
        # generating initialized gradient matrices
        layers_gradients = []
        biases_gradients = []
        for i in range(self.network_depth):
            layer = np.zeros((self.network_architecture[i], self.network_architecture[i + 1]))
            layers_gradients.append(layer)
            bias = np.zeros(self.network_architecture[i + 1])
            biases_gradients.append(bias)
        return layers_gradients, biases_gradients

    def evaluation(self, X, Y):
        # loss and accuracy evaluation on a given data and labels
        total_loss = 0.0
        n = len(Y)
        truely_predicted = 0
        for x, y in zip(X, Y):
            h_list, loss = self._forward(x, y)
            if np.argmax(h_list[-1]) == np.argmax(y):
                truely_predicted += 1
            total_loss += loss
        return total_loss / n, truely_predicted / n

    # def _relu(self, x):
    #     # relu activation function
    #     return 1.0 / (1.0 + np.exp(-x))
    #
    # def _relu_dervative(self, x):
    #     # relu derive function
    #     return x * (1.0 - x)

    def _relu(self, x):
        # relu activation function
        return np.maximum(0, x)

    def _relu_dervative(self, x):
        # relu derive function
        y = np.copy(x)
        y[y > 0.0] = 1.0
        y[y <= 0.0] = 0.00
        return y

    def predict(self, X):
        # predict new data's label
        n = len(X)
        predictions = np.zeros(n, dtype=int)
        for i, x in enumerate(X):
            h_list = self._forward(x)
            predictions[i] = np.argmax(h_list[-1])
        return predictions
