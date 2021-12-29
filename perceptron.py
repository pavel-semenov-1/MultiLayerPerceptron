import sys
import numpy as np
import random

from utils import *


class MultiLayerPerceptron:

    def compute_accuracy(self, inputs, targets):
        # Computes *classification* accuracy - percentage of correctly categorized inputs
        return np.mean([d.argmax() == self.compute_output(self.add_bias(x)).argmax() for (x, d) in zip(inputs.T, targets.T)])

    def add_bias(self, x):
        # Add bias to input vector x.
        return np.concatenate((x, [1]))

    def initialize_weights(self):
        # Sets all weights to (Gaussian) random values
        self.W = np.random.randn(self.output_dim.size, self.input_dim.size + 1)

    def sigmoid(self, x):
        # Activation function - logistical sigmoid
        return 1/(1 + np.exp(-x))

    def compute_output(self, x):
        # Computes output (vector y) of the neural network for given input vector x (including bias).
        f = self.sigmoid
        self.add_bias(x)
        return np.array([f(sum([self.W[i][j] * x[j] for j in range(input_dim+1)])) for i in range(self.output_dim)])

    def compute_error(self, d, y):
        # Computes square error of output y against desired output d.
        return sum([(d[i] - y[i])**2 for i in range(output_dim)])

    def train(self, inputs, targets, num_epochs, alpha=0.1):
        # Trains the neural network, iterating num_epochs times.
        # After each epoch, per-epoch regression error (E) and classification
        # accuracy are appended into history, that is return for further plotting.
        count = inputs.shape[1] # number of input-target pairs
        err_history = []
        accuracy_history = []

        for ep in range(num_epochs):
            E = 0
            for i in np.random.permutation(count):
                x = self.add_bias(inputs[:, i])
                d = targets[:, i]
                y = self.compute_output(x)
                e = self.compute_error(d, y)
                E += e
                self.W = self.W + alpha * (np.outer(np.array([(d[a] - y[a])*y[a]*(1-y[a]) for a in range(output_dim)]), x))

            err_history.append(E)
            acc = self.compute_accuracy(inputs, targets)
            accuracy_history.append(acc)
            if (ep+1) % 10 == 0: print('Epoch {:3d}, E = {:6.3f}, accuracy = {:4.1%}'.format(ep+1, E, acc))
        return (err_history, accuracy_history)

    def evaluate(self, file):
        data = np.loadtxt(file)
        self.input_dim, self.output_dim = data[:,:2], data[:,2:]
        self.initialize_weights()



if __name__ == "__main__":
    ## Load data and initialize
    file_path = 'mlp_train.txt'

    model = MultiLayerPerceptron()
    model.evaluate(file_path)

    ## Train the neural network


    ## Quick numpy tutorial:

##    print('Vector:')
##    a = np.array([1, 2, 3]) # vector
##    b = np.array([4, 5, 6]) # another vector
##    print(a, b)
##    print(a.shape)   # 'shape'=size of vector is tuple!
##    print(a + 100)   # vector + scalar = vector
##    print(a * 100)   # vector * scalar = vector
##    print(a ** 2)    # vector ** scalar = vector
##    print(np.exp(a)) # numpy function applies to every element of vector automatically
##    print(a + b)     # element-wise plus
##    print(a * b)     # element-wise multiplication
##    print(a.dot(b))     # dot product of vectors
##    print(np.dot(a, b)) # the same dot product of vectors
##    print(a @ b)        # the same dot product of vectors
##    print(np.outer(a, b)) # outer product of vectors
##
##    print('Matrix:')
##    P = np.array([[1, 2], [3, 4], [5, 6]]) # matrix (of size 3 rows X 2 columns)
##    R = np.array([[9,8], [7,6]])
##    print('Matrix P:\n{}\nShape of P: {}\n'.format(P, P.shape))
##    print('Matrix R:\n{}\nShape of R: {}\n'.format(P, R.shape))
##    print(P.dot(R))     # matrix multiplication
##    print(np.dot(P, R)) # the same matrix multiplication
##    print(P @ R)        # the same matrix multiplication
##    # print(np.dot(R, P)) # dimensions do not match, matrix multiplication will raise an error
##    print(a @ P)        # vector * matrix (classic dot multiplication)
##
