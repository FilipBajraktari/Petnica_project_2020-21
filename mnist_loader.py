from keras.datasets import mnist
import numpy as np

def load_data():
    training_data, test_data = mnist.load_data()
    return (training_data, test_data)

def load_data_wrapper():
    tr_d, te_d = load_data()
    #ispravljamo trening slike kako bi bile pogodne za nasu mrezu
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    #ispravljamo test slike kako bi bile pogodne za nasu mrezu
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return (training_data, test_data)

def vectorized_result(j):
    ve = np.zeros((10,1))
    ve[j] = 1.0
    return ve