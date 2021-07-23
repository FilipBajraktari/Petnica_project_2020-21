import neuralNetwork3

tr_d, val_d, te_d = neuralNetwork3.load_data_shared()
mini_batch_size = 10

net = neuralNetwork3.Network([
    neuralNetwork3.FullyConnectedLayer(n_in=784, n_out=100),
    neuralNetwork3.SoftmaxLayer(n_in=100, n_out=10)],
    mini_batch_size
)

net.SGD(tr_d, 60, mini_batch_size, 0.1, val_d, te_d)