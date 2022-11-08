import numpy as np

from Networks.Backpropagation import Backpropagation, get_circle_examples


def main():
    np.set_printoptions(suppress=True)

    network = Backpropagation(
        shape=(2, 20, 20, 20, 20, 2),
        activations=(None, 2, 2, 2, 2, 1),
    )

    label, target = get_circle_examples(500)
    network.train(label, target, 100)
    network.visualize_network()


if __name__ == '__main__':
    main()
