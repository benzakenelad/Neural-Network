import numpy as np


class NN():
    def __init__(self, network_architecture, weights_epsilon=0.1):
        self.layers = []
        self.biases = []
        self.network_depth = len(network_architecture) - 1
        self.network_architecture = network_architecture
        for i in range(self.network_depth):
            layer = self._generate_layer(network_architecture[i], network_architecture[i + 1], weights_epsilon)
            self.layers.append(layer)
            bias = self._generate_bias(network_architecture[i + 1], weights_epsilon)
            self.biases.append(bias)
        self.output_len = network_architecture[-1]

    def _forward(self, x, y=None):
        h = np.copy(x)
        h_list = [h]
        for i, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            if i + 1 != self.network_depth:
                h = self._sigmoid(np.dot(layer.T, h) + bias)
            else:
                h = self._sigmoid(np.dot(layer.T, h) + bias)
            h_list.append(h)
        if y is not None:
            loss = self._loss(h_list[-1], y)
            return h_list, loss
        return h_list

    def _backpropagation(self, h_list, y):
        errors_list = []
        h_outout_layer = h_list.pop(-1)
        error = (h_outout_layer - y) * self._sigmoid_dervative(h_outout_layer)
        for h, W in zip(reversed(h_list), reversed(self.layers)):
            errors_list.append(error)
            error = np.dot(W, error) * self._sigmoid_dervative(h)
        loss_dervative_by_W_list = []
        loss_dervative_by_b_list = []
        for h, delta in zip(h_list, reversed(errors_list)):
            dLdW = np.dot(h.reshape(-1, 1), delta.reshape(-1, 1).T)
            dLdB = delta
            loss_dervative_by_W_list.append(dLdW)
            loss_dervative_by_b_list.append(dLdB)

        return loss_dervative_by_W_list, loss_dervative_by_b_list

    def _relu(self, x):
        return np.maximum(0.01 * x, x)

    def _relu_dervative(self, x):
        y = np.copy(x)
        y[y > 0.0] = 1.0
        y[y <= 0.0] = 0.01
        return y

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_dervative(self, x):
        return x * (1.0 - x)

    def _loss(self, y_hat, y_true):
        J = np.sum((-1.0) * (y_true * np.log(y_hat) + (1.0 - y_true) * np.log(1.0 - y_hat)))
        return J

    def _softmax(self, x):
        temp = np.exp(x)
        return temp / np.sum(temp)

    def _generate_layer(self, input_len, output_len, weights_epsilon):
        return (np.random.rand(input_len, output_len) * 2 * weights_epsilon) - weights_epsilon

    def _generate_bias(self, output_len, weights_epsilon):
        return (np.random.rand(output_len) * 2 * weights_epsilon) - weights_epsilon

    def _generate_onehot(self, labels):
        n = len(labels)
        onehot_matrix = np.zeros((n, self.output_len))
        for i in range(n):
            onehot_matrix[i, labels[i]] = 1.0
        return onehot_matrix

    def _generate_initialized_gradient_matrices(self):
        layers_gradients = []
        biases_gradients = []
        for i in range(self.network_depth):
            layer = np.zeros((self.network_architecture[i], self.network_architecture[i + 1]))
            layers_gradients.append(layer)
            bias = np.zeros(self.network_architecture[i + 1])
            biases_gradients.append(bias)
        return layers_gradients, biases_gradients

    def evaluation(self, X, Y):
        total_loss = 0.0
        n = len(Y)
        truely_predicted = 0
        for x, y in zip(X, Y):
            h_list, loss = self._forward(x, y)
            if np.argmax(h_list[-1]) == np.argmax(y):
                truely_predicted += 1
            total_loss += loss
        return total_loss / n, truely_predicted / n

    def predict(self, X):
        n = len(X)
        predictions = np.zeros(n, dtype=int)
        for i, x in enumerate(X):
            h_list = self._forward(x)
            predictions[i] = np.argmax(h_list[-1])
        return predictions

    def fit(self, X, Y, epochs=1024, batch_size=32, eta=0.1, decay=0.0001, print_evaluation=False):
        n = len(Y)
        Y = self._generate_onehot(Y)
        n_batch = int(n / batch_size) + 1
        for i in range(epochs):
            perm = np.random.permutation(n)
            X, Y = X[perm], Y[perm]
            for batch_idx in range(n_batch):
                layers_gradients, biases_gradients = self._generate_initialized_gradient_matrices()
                batch_data = X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                batch_targets = Y[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                for x, y in zip(batch_data, batch_targets):
                    outputs_list, loss = self._forward(x, y)
                    dLdW, dLdb = self._backpropagation(outputs_list, y)
                    for dW, new_dW in zip(layers_gradients, dLdW):
                        dW += new_dW
                    for db, new_db in zip(layers_gradients, dLdb):
                        db += new_db
                for dW, db in zip(layers_gradients, biases_gradients):
                    dW /= batch_size
                    db /= batch_size
                for layer, dW in zip(self.layers, layers_gradients):
                    layer -= eta * (decay * layer + dW)
                for bias, dB in zip(self.biases, biases_gradients):
                    bias -= eta * dB
            if print_evaluation:
                loss, acc = self.evaluation(X, Y)
                print(f'epoch {i} train loss : {"%.2f" % loss}, train accuracy : {"%.2f" % acc}')
