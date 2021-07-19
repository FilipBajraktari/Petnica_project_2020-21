'''
neuralNetwork.py

U modulu su implementirani:
    - stochastic gradient descent
    - kvadratna cost funkcija
    - backpropagation

    - cross-entropy funkcija
    - tehnike regularizacije
    - bolja inicijalizacije tezina i sklonosti
'''
import random
import numpy as np

#aktivaciona funkcija
def sigmoid(z):
    z = z.astype('float128')
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    #prvi izvod sigmoid funkcije
    return sigmoid(z)*(1 - sigmoid(z))

class Quadractic_cost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y)*sigmoid_prime(z)

class Cross_Entropy_Cost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)
                        -(1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z, a, y):
        return (a-y)

class Network(object):
    
    def __init__(self, sizes, cost = Cross_Entropy_Cost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer(sizes)
        self.cost = cost

    def default_weight_initializer(self, sizes):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def large_weight_initializer(self, sizes):
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w, b in  zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)

        return a

    #eta is learning rate
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0, test_data=None):

        training_data = list(training_data)
        test_data = list(test_data)

        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        maks = 0
        for j in range(epochs):
            random.shuffle(training_data)

            #deklaracija mini_batch
            mini_batches = [
                training_data[k: k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
            
            if test_data:
                evaluation = self.evaluate(test_data)
                print("Epoch {0}: {1} / {2}"
                    .format(j, evaluation, n_test))
                maks = max(maks, evaluation)
            else:
                print("Epoch {0} complete".format(j))

        return maks

    #promena tezina i sklonosti
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        #parcijalni izvod za svaki bias posebno
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #parcijalni izvod za svaku tezinu posebno
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #konacan izvod minibatch-a uzimamo aritmeticku sredinu svi izvoda, zato
        #nam je potrebno da izracunamo sumu
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #abdejtovanje sklonosti i tezina
        self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]

    #nalazenje parcijalnih izvoda za sve tezine i sklonosti
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        z = activation
        zs = []
        #feedforward
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for x, y in test_data]

        return sum(int(x==y) for x, y in test_results)

    def cost_derivate(self, output_activations, y):
        return (output_activations - y)