import numpy as np

# this is tha base layer
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

#Fully connected layer without activations.
class FullyConnectedLayerWithoutActivations(Layer):
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
        d_bias = output_error

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * d_bias

        return input_error

# activation layer
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

# fully connected layer with activation
class FullyConnectedLayerWithActivation(Layer):
    def __init__(self, input_size, output_size, activation, activation_prime):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1,output_size)-0.5
        self.net = None
        self.activation = activation
        self.activation_prime = activation_prime

    def __repr__(self):
        return "w" + str(self.weights) + "\nb" + str(self.biases) + "\na" + str(self.activation)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.input_net = np.dot(self.input, self.weights) + self.biases
        self.output = self.activation(self.input_net)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        output_error_activation = self.activation_prime(self.input_net)*output_error
        input_error = np.dot(output_error_activation, self.weights.T)
        weights_error = np.dot(self.input.T, output_error_activation)
        d_bias = output_error_activation

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * d_bias
        # print("fcwaie:")
        # print(input_error)
        # print( "\n") 
        return input_error