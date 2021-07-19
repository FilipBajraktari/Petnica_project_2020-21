from bookImplementation import network, network2
import mnist_loader as ml
import neuralNetwork
import neuralNetwork2
import numpy as np
import random

sizes = [784, 30, 10]

tr_d, te_d = ml.load_data_wrapper()
net = neuralNetwork2.Network(sizes, neuralNetwork2.Cross_Entropy_Cost)
print(net.SGD(tr_d, 40, 10, 0.0005, 0.0, te_d))