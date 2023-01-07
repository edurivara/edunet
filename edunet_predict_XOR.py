import numpy as np
import matplotlib.pyplot as plt
import math_formulas

from network import Network
import layers

x_train = np.array([ [[0,0]], [[0,1]], [[1,0]], [[1,1]] ])
y_train = np.array([ [[0]],   [[1]],   [[1]],   [[0]] ])

# set seed to a number to obtain repeatable results ie: 1 for dove shaped XOR contour space
# ie: Resutls for 10000 epochs, seed(1)
# For input: [0 0]  Expected: [0]  Prediction: [0.00516257]
# For input: [0 1]  Expected: [1]  Prediction: [0.9882216]
# For input: [1 0]  Expected: [1]  Prediction: [0.98857999]
# For input: [1 1]  Expected: [0]  Prediction: [0.01733293]
np.random.seed()

net = Network()
net.set_loss_function(math_formulas.mse, math_formulas.mse_prime)

# # Separated FC and Activation layers
# net.add(layers.FullyConnectedLayerWithoutActivations(2, 3))
# net.add(layers.ActivationLayer(math_formulas.sigmoid, math_formulas.sigmoid_prime))
# net.add(layers.FullyConnectedLayerWithoutActivations(3, 1))
# net.add(layers.ActivationLayer(math_formulas.sigmoid, math_formulas.sigmoid_prime))

# Fully connected layers with activations
net.add(layers.FullyConnectedLayerWithActivation(2, 3, math_formulas.sigmoid, math_formulas.sigmoid_prime))
net.add(layers.FullyConnectedLayerWithActivation(3, 1, math_formulas.sigmoid, math_formulas.sigmoid_prime))

net.train(x_train, y_train, epochs=10000, learning_rate=0.5)

predictions = net.predict(x_train)
for index, input in enumerate(x_train):
    print("For input: " + str(input[0]) + "  Expected: " + str(y_train[index][0]) + "  Prediction: " + str(predictions[index][0]))

# Plot 3d contour of XOR prediction
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
for x in np.arange(1, step=0.05):
    for y in np.arange(1, step=0.05):
        x_plot = np.array([ [[x,y]] ])
        z = net.predict(x_plot)
        ax.scatter(x, y, z, c="c", s=50, alpha=0.5)
plt.show()