from typing import List, Callable, Tuple
from Networks.ActivationFunctions import *

import numpy as np

Int = int
Float = float
Matrix = List[List[Float]] | np.array
Vector = List[Float] | np.array

activation_functions = {
    0: (tanh, tanh_dx),
    1: (sigmoid, sigmoid_dx),
    2: (ReLu, ReLu_dx),
    3: (LReLu, LReLu_dx),
    4: (fast_tanh, fast_tanh_dx),
    None: (None, None)
}


class Network:
    def __init__(self, shape: Tuple[Int], activations: Tuple[Int]):
        """
        :param shape: The shape of the network. The first element is the size of the input layer, the last element
        is the size of the output layer.
        :param activations: The activation functions (AF) for each layer. For the input layer you don't have
        to provide an AF, cause no function is applied on the input.
        """
        assert len(shape) == len(activations), 'The shape of the network mismatches the number of activations'

        Layer.layer_index_counter = 0

        self.shape = shape
        self.activations = activations
        self.layers = [Layer(size, activation, self) for size, activation in zip(self.shape, self.activations)]

    def forward_propagation(self, inputs: List[Vector]) -> List[Vector]:
        """  Propagates a list of input vector through the network.

        :param inputs: A list of vectors. For N vectors, the size of the input list is: self.shape[0] * N
        :return: The output of the network. For N vectors, the size of the output List is: self.shape[-1] * N
        """
        self.layers[0].val = np.array(inputs)
        for layer in self.layers[1:]:  # skip input layer
            layer.entry = layer.W.dot(self.layers[layer.index - 1].val.T).T + layer.B
            layer.val = layer.activation(layer.entry)
        return self.layers[-1].val

    def get_network_values(self) -> List[Float]:
        """ Converts the network parameters into a unique float list which can be saved and then loaded with the
        set_network_values function.

        :return: A list of floats i.e. the DNA of the Network.
        """
        genome: List[Float] = []
        for layer in self.layers[1:]:
            genome = np.concatenate((genome, layer.B, layer.W.flatten()))
        return genome

    def set_network_values(self, genome: np.array) -> None:
        """ reconstructs a network from a given 1d float array

        :param genome: A list of floats i.e. the DNA of the Network.
        """
        pointer: Int = 0
        for layer in self.layers[1:]:
            layer.B = genome[pointer:pointer + layer.size]
            pointer += layer.size

            s1, s2 = layer.w_shape
            layer.W = genome[pointer:pointer + s1 * s2].reshape(s1, s2)
            pointer += s1 * s2

    def visualize_network(self, *, resolution: Int = 400) -> None:
        """ Creates a 2d visualisation what the network predicts the output should be in the (0, 1) interval.

        :param resolution: The resolution of the visualisation.
        """

        import matplotlib.pyplot as plt
        lst = []
        for i in np.linspace(0, 1, resolution):
            ma = np.linspace((i, 0), (i, 1), resolution)
            result = [x[0] for x in self.forward_propagation(ma)]
            lst.append(result)

        plt.imshow(lst, cmap='magma')
        plt.show()


class Layer:
    weights_range = (-1, 1)
    bias_range = (0, 1)
    layer_index_counter = 0

    def __init__(self, size: Int, activation: Callable, net_work: Network):
        """
        :param size: The number of neurons in the layer
        :param activation: The activation function used for the layer. None if it is the input layer
        :param net_work: The network
        """
        self.network: Network = net_work
        self.index: Int = Layer.layer_index_counter
        self.size: Int = size
        self.w_shape: Tuple[Int, Int] = (self.size, self.network.shape[self.index - 1])

        self.activation: Callable
        self.activation_dx: Callable
        self.activation, self.activation_dx = activation_functions.get(activation)

        self.val: Vector = np.zeros(self.size)
        self.entry: Vector = np.zeros(self.size)
        self.error_signal: Vector = np.zeros(self.size)

        self.B_delta: Vector = np.zeros(self.size)
        self.W_delta: Vector = np.zeros(shape=self.w_shape)

        self.B: Vector
        self.W: Matrix
        if self.index:  # if the layer is not the input layer create weights and biases for the layer
            # initial weights range based in xavier initialisation
            self.xavier_range = np.array([-1, 1]) * np.sqrt(6) / np.sqrt(self.size + self.network.shape[self.index - 1])
            self.B = np.random.uniform(*Layer.bias_range, self.size)

            # use the Kaiming initialisation for layers with a non symmetric about the origin
            self.W = np.random.uniform(*Layer.weights_range, size=self.w_shape)

        Layer.layer_index_counter += 1
