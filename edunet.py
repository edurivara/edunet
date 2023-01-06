import numpy as np
np.set_printoptions(precision=10)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def mse(y_true, y_pred):
    return np.mean(np.power((y_pred - y_true),2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def __repr__(self):
        layersstr = ""
        for layer in self.layers:
            layersstr += str(layer) + "\n\n"
        return layersstr

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                #forward_propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                #err used for display purposes
                err += self.loss(y_train[j], output)

                #backpropagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            #calculate average error in all samples for this epoch
            err /= samples
            print('epoch %d/%d  error=%f' % (i+1, epochs, err))

    def predict(self, input_data):
        samples = len(input_data)
        results = []

        #run network over all samples
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            results.append(output)

        return results

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

class FCLayer(Layer):
    def __init__(self, input_size, output_size):

        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1,output_size)-0.5

    def __repr__(self):
        return "w" + str(self.weights) + "\nb" + str(self.biases)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        #dbias = output_error

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def __repr__(self):
        return "a" + str(self.activation)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    #learning_rate not used in activation
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

x_train = np.array([ [[0,0]], [[0,1]], [[1,0]], [[1,1]] ])
y_train = np.array([ [[0]],   [[1]],   [[1]],   [[0]] ])

net = Network()
net.use(mse, mse_prime)
net.add(FCLayer(2,3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3,1))
net.add(ActivationLayer(tanh, tanh_prime))

net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)
print(net.predict(x_train))
