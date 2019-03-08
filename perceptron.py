from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
    def __init__(self, n_inputs, save_fig=False):
        self.__weights = np.array([0.0] * (n_inputs + 1)) # 1 more for bias
        self.__save_fig = save_fig

    @property
    def weights(self):
        return self.__weights[:]

    def update_weights(self, X:list, y:int):
        print('>> Updating')

        self.__weights += X*y   # w += delta_w --> y * X
        print('after update: {}'.format(self.__weights))

    def activation(self, y): # sign
        if y > 0:
            return 1
        else:
            return -1

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

        # check_error_results
        result = self.check_error(datasets, n_iteration)

        while result:
            n_iteration += 1
            X, target = result
            self.update_weights(X, target)

            plot_data_and_line(datasets, W=self.weights, iter_time=n_iteration, save_fig=self.__save_fig)
            result = self.check_error(datasets, n_iteration)

def plot_data_and_line(pd_data, W, iter_time='final', save_fig=False):
    fig = plt.gcf()
    ax = fig.add_subplot(111)

    # plot data
    pd_data[pd_data['target']==-1].plot.scatter(x='sepal length (cm)', y='petal length (cm)', s = 80, c = 'b', marker = "o", ax=ax)
    pd_data[pd_data['target']==1].plot.scatter(x='sepal length (cm)', y='petal length (cm)', s = 80, c = 'r', marker = "x", ax=ax)
    fig.set_size_inches(6, 6)

    # draw line
    l = np.linspace(3,8)
    a, b = - W[1] / W[2], - W[0] / W[2]
    ax.plot(l, a*l + b, 'b-')
    plt.draw()
    plt.savefig('iteration_{}.png'.format(iter_time))
    plt.show()

def iris_data_preprocess():

    # read data
    iris = datasets.load_iris()

    # feature selection
    feature_columns = ['sepal length (cm)', 'petal length (cm)', 'target']
    target_classes = [0, 1]

    x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    y = pd.DataFrame(iris['target'], columns=['target'])
    iris_data = pd.concat([x,y], axis=1)

    iris_data = iris_data[feature_columns]
    iris_data = iris_data[iris_data['target'].isin(target_classes)]

    iris_data['target'] = iris_data['target'].map({0:-1, 1:1})

    print(iris_data.head(3))
    print(iris_data.shape)

    return iris_data

def main():
    # read data and preprocessing
    iris_data = iris_data_preprocess()

    # build model
    myPerceptron = Perceptron(n_inputs=len(iris_data.columns)-1, save_fig=False)
    myPerceptron.train(iris_data)

    # plot the result
    plot_data_and_line(iris_data, myPerceptron.weights, save_fig=False)

if __name__ == '__main__':
    main()
