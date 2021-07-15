import mnist_loader as ml
import neuralNetwork
import numpy as np
import random

def sigmoid(z):
    z = z.astype('float128')
    return 1.0/(1.0 + np.exp(-z))
    
def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))

def cost_derivate(output_activations, y):
    return (output_activations - y)

tr_d, te_d = ml.load_data_wrapper()
tr_d = list(tr_d)
print(len(tr_d))
te_d = list(te_d)
sizes = [784, 30, 10]
num_layers = 3
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

nabla_w = [np.zeros(w.shape) for w in weights]
nabla_b = [np.zeros(b.shape) for b in biases]

mini_batch = tr_d[0:10]
print(len(mini_batch))

x = tr_d[0][0]
activation = x
activations = [x]
z = activation
zs = []
#feedforward
for w, b in zip(weights, biases):
    z = np.dot(w, activation) + b
    zs.append(z)
    activation = sigmoid(z)
    activations.append(activation)

delta = cost_derivate(activations[-1], tr_d[0][1])*sigmoid_prime(zs[-1])
#print(np.array(np.dot(delta, activations[-2].transpose())).shape)
