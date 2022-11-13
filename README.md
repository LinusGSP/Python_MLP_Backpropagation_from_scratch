# Multilayer perceptron from scratch
![](https://github.com/LinusGSP/Python_MLP_Backpropagation_from_scratch/blob/master/figures/1656436889_979990_(2%2C%2024%2C%2024%2C%2024%2C%2012%2C%2012%2C%202)_(None%2C%200%2C%202%2C%202%2C%202%2C%202%2C%201).gif)
|:--:|
|*Simple Classification using genetic optimization*|
# Project overview:
The project is an implementation of a multilayer perceptron.
The layers, network and activation functions are all implemented from scratch using only numpy.

## Optimisation techniques
The project implements 2 optimization techniques:

1. Standart backpropagation using the stochastic gradient descent algorithm.
2. An experimental Genetic aproach.

Both methods are currently functional, but both still have a lot of room for improvement.

## Activation functions
The project implements 5 activation functions and their derivatives:

1. Sigmoid
2. ReLU
3. Leaky ReLU
4. Tanh
5. Tanh (approximated)


## Sources

- [Standford, Additional Notes on Backpropagation ](https://cs229.stanford.edu/notes-spring2019/backprop.pdf)
- [Backpropagation ](https://en.wikipedia.org/wiki/Backpropagation)


## Figures
||
|:--:|
![](https://github.com/LinusGSP/Python_MLP_Backpropagation_from_scratch/blob/master/figures/1656439336_715418_(2%2C%2024%2C%2024%2C%2024%2C%2012%2C%2012%2C%202)_(None%2C%200%2C%202%2C%202%2C%202%2C%202%2C%201).gif)
|*Classification using genetic optimization, log scale*|
![](https://github.com/LinusGSP/Python_MLP_Backpropagation_from_scratch/blob/master/figures/11260%5B2%2C%206%2C%204%2C%202%5D%5BNone%2C%202%2C%202%2C%202%2C%200%5D.gif)
|*Classification using a small network with gradient descent*|
![](https://github.com/LinusGSP/Python_MLP_Backpropagation_from_scratch/blob/master/figures/369941649349996.6259873%5B2%2C%206%2C%206%2C%206%2C%204%2C%202%5D%5BNone%2C%202%2C%202%2C%202%2C%200%2C%200%5D.gif)
|*Classification using gradient descent*|
![](https://github.com/LinusGSP/Python_MLP_Backpropagation_from_scratch/blob/master/figures/29793%5B2%2C%206%2C%204%2C%202%2C%201%5D.gif)
|*Early version not optimized, gradient decent*|
![](https://github.com/LinusGSP/Python_MLP_Backpropagation_from_scratch/blob/master/figures/early_version_error.png)
|*Early version, plot of multiple trainings*|
![](https://github.com/LinusGSP/Python_MLP_Backpropagation_from_scratch/blob/master/figures/eiffel_tower.png)
|*Image regression with genetic optimization (original left, learned right)*|
