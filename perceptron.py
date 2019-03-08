from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
    def __init__(self, n_inputs):
        # self.__inputs = []
        # self.__output = 0
        self.__weights = [0.0] * (n_inputs + 1) # bias

    @property
    def weights(self):
        return self.__weights[:]

    def update_weights(self, X:list, y:int):
        print('updating')

        # w = w + delta_w
        # delta_w = y * X
        X  = X[:]
        self.__weights = [self.__weights[i] + X[i] * y for i in range(len(self.__weights))]
        print('after update: {}'.format(self.__weights))

    def activation(self, y): # sign
        if y > 0:
            return 1
        else:
            return -1
    def check_error(self, datasets):
        n_weights = len(self.__weights)
        error = 0
        result = False
        for i in range(datasets.shape[0]):
            row = datasets.iloc[i].tolist()
            X, target = [1] + row[:-1], row[-1]
            if target == 0:
                target = -1
            # print('# {}, X = {}, target = {}'.format(i, X, target))

            y = sum([self.__weights[j] * X[j] for j in range(n_weights)])
            ay = self.activation(y)
            # print('predict y = {}'.format(ay))
            if target != ay:
                # self.update_weights(X, target)
                error +=1
                result = X, target
        print('#{}: error = {}'.format(self.counter, error))
        return result

    def train(self, datasets):
        n_weights = len(self.__weights)
        if len(datasets.columns) != n_weights:
            raise Exception("Wrong inputs of training!")

        self.counter = 0
        result = self.check_error(datasets)
        # plot_data_and_line(datasets, self.weights)
        while result:
            self.counter +=1
            X, target = result
            self.update_weights(X, target)
            plot_data_and_line(datasets, self.weights, counter=self.counter)
            result = self.check_error(datasets)

def plot_data_and_line(pd_data, w, counter='final'):
    fig = plt.gcf()
    ax1 = fig.add_subplot(111)

    # plot data
    pd_data[pd_data['target']==0].plot.scatter(x='sepal length (cm)', y='petal length (cm)', s = 80, c = 'b', marker = "o", ax=ax1)
    pd_data[pd_data['target']==1].plot.scatter(x='sepal length (cm)', y='petal length (cm)', s = 80, c = 'r', marker = "x", ax=ax1)
    fig.set_size_inches(6, 6)

    # draw line
    # w = myPerceptron.weights
    l = np.linspace(3,8)
    a,b = -w[1]/w[2], -w[0]/w[2]
    ax1.plot(l, a*l + b, 'b-')
    # plt.legend(loc='upper left')
    plt.draw()
    plt.savefig('iteration_{}.png'.format(counter))
    plt.show()
    

def main():
    # read data
    iris = datasets.load_iris()
    x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    y = pd.DataFrame(iris['target'], columns=['target'])
    iris_data = pd.concat([x,y], axis=1)
    iris_data = iris_data[['sepal length (cm)', 'petal length (cm)', 'target']]
    iris_data = iris_data[iris_data['target'].isin([0,1])]
    print(iris_data.head(10))
    print(iris_data.shape)

    # build model
    myPerceptron = Perceptron(n_inputs=len(iris_data.columns)-1)
    myPerceptron.train(iris_data)

    # plot the result
    plot_data_and_line(iris_data, myPerceptron.weights)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    #
    # # plot data
    # iris_data[iris_data['target']==0].plot.scatter(x='sepal length (cm)', y='petal length (cm)', s = 80, c = 'b', marker = "o", ax=ax1)
    # iris_data[iris_data['target']==1].plot.scatter(x='sepal length (cm)', y='petal length (cm)', s = 80, c = 'r', marker = "x", ax=ax1)
    # fig.set_size_inches(6, 6)
    #
    # # draw line
    # w = myPerceptron.weights
    # l = np.linspace(3,8)
    # a,b = -w[1]/w[2], -w[0]/w[2]
    # ax1.plot(l, a*l + b, 'b-')
    # plt.legend(loc='upper left')
    # plt.show()


if __name__ == '__main__':
    main()
