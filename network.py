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
            if ((i+1)%100==0):
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
