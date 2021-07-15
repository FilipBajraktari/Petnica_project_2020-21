import random
import numpy as np

class Network(object):
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w, b in  zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)

        return a

    #eta is learning rate
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        test_data = list(test_data)
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            #deklaracija mini_batch
            mini_batches = [
                training_data[k: k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(5))

    #promena tezina i sklonosti
    def update_mini_batch(self, mini_batch, eta):
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
        self.weights = [w - (eta/len(mini_batch))*nw
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

        delta = self.cost_derivate(activations[-1], y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_Data):
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for x, y in test_Data]
        return sum(int(x==y) for x, y in test_results)

    def cost_derivate(self, output_activations, y):
        return (output_activations - y)

#aktivaciona funkcija
def sigmoid(z):
    z = z.astype('float128')
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    #prvi izvod sigmoid funkcije
    return sigmoid(z)*(1 - sigmoid(z))