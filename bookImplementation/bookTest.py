import pickle
import gzip
import network3

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = network3.Network([
      network3.FullyConnectedLayer(n_in=784, n_out=100),
      network3.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data)