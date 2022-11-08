from typing import List, Tuple

import numpy as np
from Networks.Network import Network, Layer

Int = int
Float = float
Matrix = List[List[Float]] | np.array
Vector = List[Float] | np.array


class Backpropagation(Network):
    def __init__(self, shape: Tuple[Int], activations: Tuple[Int],
                 learning_rate: Float = 0.01, batch_size: Int = 1, momentum_rate: Float = 0.0,
                 l1_regularization_rate: Float = 0.0, l2_regularization_rate: Float = 0.0):
        """
        :param shape: The shape of the network. The first element is the size of the input layer, the last element
        is the size of the output layer.
        :param activations: The activation functions (AF) for each layer. For the input layer you don't have
        to provide an AF, cause no function is applied on the input.
        :param learning_rate: The rate
        :param batch_size: The batch size of the network.
        :param momentum_rate: The momentum rate of the network.
        :param l1_regularization_rate: The L1 regularization rate of the network.
        :param l2_regularization_rate: The L2 regularization rate of the network.
        """
        super().__init__(shape, activations)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum_rate = momentum_rate
        self.l1_regularization_rate = l1_regularization_rate
        self.l2_regularization_rate = l2_regularization_rate
        self.loss_progress = []

    def backpropagation(self, labels: List[Vector], targets: List[Vector]) -> Float:
        """ Calculate the error signal and the bias, weights errors.

        :param labels: The label values. For N vectors, the size of the labels is: self.shape[0] * N
        :param targets: The target values. For N vectors, the size of the target is: self.shape[-1] * N
        :return: The error of the network for the given labels and targets
        """
        output: List[Vector] = self.forward_propagation(labels)
        last_layer: Layer = self.layers[-1]
        last_layer.error_signal = (output - targets) * last_layer.activation_dx(last_layer.entry)

        for layer in reversed(self.layers[1:-1]):
            next_layer: Layer = self.layers[layer.index + 1]
            layer.error_signal = next_layer.W.T.dot(next_layer.error_signal.T).T * layer.activation_dx(layer.entry)

        for layer in self.layers[1:]:
            layer.W_delta += layer.error_signal.T.dot(self.layers[layer.index - 1].val)
            layer.B_delta += np.sum(layer.error_signal, axis=0)

        return np.average(np.linalg.norm(output - targets, axis=-1))

    def update_weights(self) -> None:
        """ Update the weights and biases of the network. """
        for layer in self.layers[1:]:
            layer.B -= (layer.B_delta * self.learning_rate) / self.batch_size
            layer.W -= (layer.W_delta * self.learning_rate) / self.batch_size
            # self.momentum(layer)
            layer.B_delta = np.zeros(layer.B.shape)
            layer.W_delta = np.zeros(layer.W.shape)

    def momentum(self, layer) -> None:
        """ update the weights and biases of the network with momentum """
        layer.B_delta = self.momentum_rate * layer.B_delta + (1 - self.momentum_rate) * layer.B_delta
        layer.W_delta = self.momentum_rate * layer.W_delta + (1 - self.momentum_rate) * layer.W_delta

    def l1_regularization(self) -> None:
        """ apply l1 regularization to the network """
        for layer in self.layers[1:]:
            layer.W -= self.learning_rate * self.l1_regularization_rate * np.sign(layer.W)

    def l2_regularization(self) -> None:
        """ apply l2 regularization to the network """
        for layer in self.layers[1:]:
            layer.W -= self.learning_rate * self.l2_regularization_rate * np.sum(layer.W)

    def train(self, labels: List[Vector], targets: List[Vector], epochs: Int) -> None:
        """ Train the network for a given number of epochs.

        :param labels: The label values. For N vectors, the size of the labels is: self.shape[0] * N
        :param targets: The target values. For N vectors, the size of the target is: self.shape[0] * N
        :param epochs: The number of training steps.
        """
        error_lst = []
        for epoch in range(epochs):
            error = 0
            for i in range(0, len(labels), self.batch_size):
                error += self.backpropagation(labels[i:i + self.batch_size], targets[i:i + self.batch_size])
                self.update_weights()
            print(f"Epoch {epoch + 1}/{epochs} - Error: {error / (len(labels) / self.batch_size)}")
            error_lst.append(error / (len(labels) / self.batch_size))

    def test(self, labels: List[Vector], targets: List[Vector]) -> None:
        """ test the network """
        output: List[Vector] = self.forward_propagation(labels)
        print(f"Error: {np.average(np.linalg.norm(output - targets, axis=-1))}")


def get_circle_examples(n: int = 500) -> (list[[float, float]], list[float, float]):
    """ Creates example data points. The negative values are scattered in a circle in the middle. The positive
    values are scattered around the circle in the middle

    :param n: The number of training samples
    :return:
    """
    def get_example1():
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        x, y = np.dot(rot, (0, np.random.uniform(-0.08, 0.08) + 0.4))  # -0.15, 0.15 + 0.5
        return [x + 0.5, y + 0.5], [0.1, 0.9]

    def get_example2():
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, 0.05) / 2)
        x, y = r * np.cos(theta), r * np.sin(theta)
        return [x + 0.5, y + 0.5], [0.9, 0.1]

    training_values, training_results = [], []
    for _ in range(n):
        r = np.random.uniform(0, 1)
        if r < 0.5:
            val, res = get_example1()
            training_values.append(val)
            training_results.append(res)
        else:
            val, res = get_example2()
            training_values.append(val)
            training_results.append(res)
    return training_values, training_results

