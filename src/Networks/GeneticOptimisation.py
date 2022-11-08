import random
import time
from typing import List, Callable, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

from Network import Network

Int = int
Float = float
Matrix = List[List[Float]] | np.array
Vector = List[Float] | np.array


class Net(Network):
    def __init__(self, shape: Tuple[Int], activations: Tuple[Int]):
        super().__init__(shape, activations)
        self.fitness: Float = 1

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __str__(self):
        return f'{self.fitness}'

    def mutate(self, chance: Float = 0.1) -> None:
        genome = self.get_network_values()
        random_choice = np.random.uniform(-1, 1, size=len(genome))
        threshold = chance * 2 - 1
        for i, v in enumerate(random_choice):
            if v < threshold:
                genome[i] = v
        self.set_network_values(genome)

    def get_plot_data(self, *, resolution: Int = 200) -> None:
        lst = []
        for i in np.linspace(0, 1, resolution):
            ma = np.linspace((i, 0), (i, 1), resolution)
            result = [1 - x[1] + x[0] for x in self.forward_propagation(ma)]
            lst.append(result)
        return lst


class Generative:
    def __init__(self, gen_count: Int, gen_size: Int, net: Network, fitness_function: Callable):
        self.gen_count: Int = gen_count
        self.gen_size: Int = gen_size

        self.shape: Tuple[Int] = net.shape
        self.act: Tuple[Callable] = net.activations
        self.fitness_function: Callable = fitness_function

        self.current_gen: List[Network]
        self.current_gen = [Net(shape=self.shape, activations=self.act) for _ in range(self.gen_size * 100)]

    def start(self):
        """ starts the training """
        errors = [1]
        for gen in range(self.gen_count):
            self.fitness_function(self)
            errors.append(self.current_gen[0].fitness)
            self.top_select(selection)
            print(gen, [x.fitness for x in self.current_gen])
            self.fill_up(fill)
        return errors[-1]

    def _animate(self, frames):
        if frames == self.gen_count - 1:
            return
        self.fitness_function(self)
        self.tournament_select(selection)
        print(frames, [x.fitness for x in self.current_gen])
        self.fill_up(fill)
        best = self.get_best_network()

        self.im.set_array(best.get_plot_data())
        self.plt_error.append(best.fitness)
        self.ax2.plot(self.plt_error, color="green")
        self.title.set_text(f'Generation: {frames}')
        self.text.set_text(f'Network Error: {best.fitness:.10f}')
        self.vl.set_ydata([best.fitness, best.fitness])

        return [self.im]

    def animated_start(self):
        self.plt_error = []
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))
        plt.subplots_adjust(right=0.9)
        self.im = self.ax1.imshow(np.random.random((200, 200)), cmap="magma", extent=[0, 1, 0, 1])
        self.im.set_clim(0, 1)
        self.plt_error = []
        self.ax1.set_xticks(np.arange(0, 3, 1))
        self.ax1.set_yticks(np.arange(0, 3, 1))
        self.ax1.set_title('Network Map')

        self.ax2.set_ylabel("Network Loss")
        self.ax2.set_xlabel(f"Generations")
        # self.ax2.set_ylim(top=0.5)
        self.ax2.set_yscale('log')
        # self.ax2.yaxis.set_minor_locator(tck.AutoMinorLocator(5))
        self.title = self.fig.suptitle(f'Training Iteration: {0}', fontsize=16)
        self.fig.colorbar(self.im, ax=self.ax1, ticks=np.linspace(-0.1, 1.1, 13, endpoint=True))
        self.text = self.ax2.set_title(f'Network Error: {1}', animated=True)
        self.vl = self.ax2.axhline(y=1, color='k')
        self.plot_data_points()

        ani = matplotlib.animation.FuncAnimation(self.fig, self._animate, frames=range(self.gen_count), repeat=False)

        ani.save(f'{int(time.time())}_{seed}_{self.shape}_{self.act}.gif', writer='ImageMagickWriter', fps=15)
        plt.show()
        print("hahahahah")

    def plot_data_points(self):
        examples = get_circle_examples(300)
        for v, r in zip(*examples):
            x, y = v
            if r == [0.1, 0.9]:
                self.ax1.scatter(x, y, color='red', alpha=0.3, s=50)
            if r == [0.9, 0.1]:
                self.ax1.scatter(x, y, color='blueviolet', alpha=0.3, s=50)

    def top_select(self, count: Int) -> None:
        """ selects the best performing networks """
        self.current_gen = sorted(self.current_gen, reverse=False)[:count]

    def tournament_select(self, count: Int):
        self.current_gen = sorted(self.current_gen)
        new_gen = self.current_gen[:2]
        while len(new_gen) < count:
            new_gen.append(min(random.sample(self.current_gen, k=3)))
        self.current_gen = new_gen

    def fill_up(self, count) -> None:
        """ fills up the current generation with new crossed over Networks"""
        for i in range(self.gen_size - len(self.current_gen)):
            new = crossover(*random.sample(self.current_gen, k=2), count=count)
            new.mutate(mutation_rate)
            self.current_gen.append(new)

    def get_best_network(self):
        return min(self.current_gen)


def crossover(net1: Net, net2: Net, count=1):
    """ produces a neu network containing part of net1 and net2 by crossover """
    new_network: Net = Net(shape=net1.shape, activations=net1.activations)

    seq1 = net1.get_network_values()
    seq2 = net2.get_network_values()

    length = len(seq1)
    crossover_points = np.concatenate((np.sort(np.random.choice(length, count, replace=False)), [length]))

    new_genome = seq1
    for i in range(count):
        if not i % 2:
            new_genome[crossover_points[i]:crossover_points[i + 1]] = seq2[crossover_points[i]:crossover_points[i + 1]]
    new_network.set_network_values(new_genome)
    return new_network


def avg_crossover(net1: Net, net2: Net, count=1):
    new_network: Net = Net(shape=net1.shape, activations=net1.activations)

    seq1 = net1.get_network_values()
    seq2 = net2.get_network_values()

    length = len(seq1)
    crossover_points = np.concatenate((np.sort(np.random.choice(length, count, replace=False)), [length]))

    new_genome = seq1
    for i in range(count):
        if not i % 2:
            new_genome[crossover_points[i]:crossover_points[i + 1]] = seq2[crossover_points[i]:crossover_points[i + 1]]
    new_network.set_network_values(new_genome)
    return new_network


def get_circle_examples(n: int = 500) -> list:
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


def fitness(generation: Generative):
    for net in generation.current_gen:
        net.fitness = calculate_error(net)


def calculate_error(net: Net) -> Float:
    train_v, train_r = training_set
    output: List[Vector] = net.forward_propagation(train_v)
    return np.average(np.linalg.norm(output - train_r, axis=-1))


training_set = get_circle_examples(500)


def main():
    np.set_printoptions(suppress=True)
    n = Net(shape=(2, 24, 24, 24, 12, 12, 2), activations=(None, 0, 2, 2, 2, 2, 1))

    gen = Generative(200, 100, net=n, fitness_function=fitness)
    n = gen.get_best_network()
    n.visualize_network()
    # gen.animated_start()


    # import cProfile
    # import pstats
    # with cProfile.Profile() as pr:
    #     for _ in range(100000):
    #         crossover(n1, n2)
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.CUMULATIVE)
    # stats.print_stats()


if __name__ == '__main__':
    selection, fill, mutation_rate = 3, 1, 0.0005  # 3, 1, 0.001 # top_select

    seed = np.random.randint(0, 999_999)
    # seed = 715418
    np.random.seed(seed)
    random.seed(seed)

    err = main()
    print(f'seed={seed}, {selection=}, {fill=}, {mutation_rate=}, {err=}')
