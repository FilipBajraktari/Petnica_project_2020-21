import mnist_loader as ml
from bookImplementation import network
import neuralNetwork
import numpy as np
import random

sizes = [784, 30, 10]

tr_d, te_d = ml.load_data_wrapper()
net = network.Network(sizes)
net.SGD(tr_d, 30, 10, 0.1, te_d)
