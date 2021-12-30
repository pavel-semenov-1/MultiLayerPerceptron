import sys
import numpy as np
import random

from utils import *


class MultiLayerPerceptron:

    def compute_accuracy(self, inputs, targets):
        # Computes *classification* accuracy - percentage of correctly categorized inputs
        return np.mean([d.argmax() == self.compute_output(self.add_bias(x)).argmax() for (x, d) in zip(inputs, targets)])

    def add_bias(self, x):
        # Add bias to input vector x.
        return np.concatenate((x, [1]))

    def initialize_weights(self):
        # Sets all weights to (Gaussian) random values
        self.W_hid = np.random.randn(self.hidden_dim, self.input_dim+1)
        self.W_out = np.random.randn(self.hidden_dim, self.output_dim)

    def sigmoid(self, x):
        # Activation function - logistical sigmoid
        return 1/(1 + np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def compute_error(self, d, y):
        # Computes square error of output y against desired output d.
        return sum([(d[i] - y[i])**2 for i in range(self.output_dim)])

    def load_data(self, file):
        data = np.loadtxt(file)
        np.random.shuffle(data)
        train_count = int(data.shape[0]*0.8)
        return data[:train_count], data[train_count:]

    def compute_output(self, x):
        f_out = self.sigmoid
        f_hid = self.sigmoid
        net_hid = [sum([self.W_hid[i][j]*x[j] for j in range(self.input_dim+1)]) for i in range(self.hidden_dim)]
        h = [f_hid(net_hid[i]) for i in range(self.hidden_dim)]
        net_out = sum([self.W_out[i]*h[i] for i in range(self.hidden_dim)])
        return f_out(net_out)

    def train(self, inputs, targets, num_epochs=100, alpha=0.1):
        count = inputs.shape[0]
        f_out = self.sigmoid
        f_hid = self.sigmoid
        f_out_der = self.sigmoid_der
        f_hid_der = self.sigmoid_der

        for ep in range(num_epochs):
            E = 0
            for index in np.random.permutation(count):
                x = self.add_bias(inputs[index])
                d = targets[index]
                net_hid = [sum([self.W_hid[i][j]*x[j] for j in range(self.input_dim+1)]) for i in range(self.hidden_dim)]
                h = [f_hid(net_hid[i]) for i in range(self.hidden_dim)]
                net_out = sum([self.W_out[i]*h[i] for i in range(self.hidden_dim)])
                y = f_out(net_out)
                e = self.compute_error(d, y)
                E += e
                delta_out = (d-y)*f_out_der(net_out)
                delta_hid = [self.W_out[i]*delta_out*f_hid_der(net_hid[i]) for i in range(self.hidden_dim)]
                for i in range(self.output_dim):
                    self.W_out[i] = self.W_out[i] + alpha*delta_out*h[i]
                for i in range(self.hidden_dim):
                    self.W_hid[i] = self.W_hid[i] + alpha*delta_hid[i]*x
            if (ep+1)%10 == 0:
                print('epoch ' + str(ep+1) + ' accuracy ' + str(self.compute_accuracy(inputs, targets)))

    def validate(self, data):
        inputs, targets = data[:,:2], data[:,2:]
        predicted = []
        for i in range(len(inputs)):
            x = self.add_bias(inputs[i])
            y = self.compute_output(x)
            predicted.append(y)
        show_data(inputs, targets, predicted=predicted)
        
    def evaluate(self, file):
        train_data, validate_data = self.load_data(file)
        inputs, targets = train_data[:,:2], train_data[:,2:]
        self.input_dim, self.output_dim, self.hidden_dim =  inputs.shape[1], targets.shape[1], inputs.shape[1]
        self.initialize_weights()
        self.train(inputs, targets)
        self.validate(validate_data)


if __name__ == "__main__":
    file_path = 'mlp_train.txt'

    model = MultiLayerPerceptron()
    model.evaluate(file_path)

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
