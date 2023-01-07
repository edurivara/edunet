import numpy as np
import matplotlib.pyplot as plt
import math_formulas

from network import Network
import layers

# np.set_printoptions(precision=10)

x_train = np.array([ [[0,0]], [[0,1]], [[1,0]], [[1,1]] ])
y_train = np.array([ [[0]],   [[1]],   [[1]],   [[0]] ])

net = Network()
net.use(math_formulas.mse, math_formulas.mse_prime)
net.add(layers.FCLayer(2,3))
net.add(layers.ActivationLayer(math_formulas.sigmoid, math_formulas.sigmoid_prime))
net.add(layers.FCLayer(3,1))
net.add(layers.ActivationLayer(math_formulas.sigmoid, math_formulas.sigmoid_prime))

net.fit(x_train, y_train, epochs=10000, learning_rate=0.5)
print(net.predict(x_train))

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
