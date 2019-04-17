from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../../ml')

from utils import plot_data_and_line
from activation.activation import ActivationFunction
from preprocess.iris_preprocess import iris_data_preprocess

class Perceptron:
    def __init__(self, n_inputs, activ_func='Sign', save_fig=False):
        self.__weights = np.array([0.0] * (n_inputs + 1)) # 1 more for bias
        self.__save_fig = save_fig
        self.__activation = ActivationFunction(activ_func)

    @property
    def weights(self):
        return self.__weights[:]

    def update_weights(self, X:list, y:int):
        print('>> Updating')

        self.__weights += X*y   # w += delta_w --> y * X
        print('after update: {}'.format(self.__weights))

    def activation(self, y): # sign
        return self.__activation.func(y)
        # if y > 0:
        #     return 1
        # else:
        #     return -1

    def check_error(self, datasets, n_iteration):
        n_weights = len(self.__weights)
        error = 0
        result = False
        for i in range(datasets.shape[0]):
            row = datasets.iloc[i].tolist()
            X, target = np.array([1] + row[:-1]), row[-1]  # initial x[0] as 1, for bias, so that w0*x0 = w0 (aka. b)

            y = self.activation(np.dot(self.__weights, X)) # W*X = w0*x0 + w1*x1 + ... + w_n*x_n

            if target != y:
                error += 1
                result = X, target
        print('Iteration #{}: error = {}'.format(n_iteration, error))
        return result

    def train(self, datasets):
        n_iteration = 0
        n_weights = len(self.__weights)
        if len(datasets.columns) != n_weights:
            raise Exception("Wrong inputs of training!")

        x_title, y_title = list(datasets.columns)[:2]
        # check_error_results
        result = self.check_error(datasets, n_iteration)

        while result:
            n_iteration += 1
            X, target = result
            self.update_weights(X, target)

            plot_data_and_line(datasets, W=self.weights, iter_time=n_iteration, x_title=x_title, y_title=y_title, save_fig=self.__save_fig)
            result = self.check_error(datasets, n_iteration)


def main():
    # read data and preprocessing
    iris_data = iris_data_preprocess()

    # build model
    myPerceptron = Perceptron(n_inputs=len(iris_data.columns)-1, save_fig=False)
    myPerceptron.train(iris_data)

    # plot the result
    x_title, y_title = list(iris_data.columns)[:2]
    plot_data_and_line(iris_data, myPerceptron.weights, x_title=x_title, y_title=y_title, save_fig=False)

if __name__ == '__main__':
    main()
