from Networks.Backpropagation import Backpropagation, get_circle_examples


def main():
    network = Backpropagation(
        shape=(2, 8, 8, 2),
        activations=(None, 2, 2, 1)
    )

    label, target = get_circle_examples(500)
    network.train(label, target, 150)
    network.visualize_network()


if __name__ == '__main__':
    main()
