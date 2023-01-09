import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils

import math_formulas
from network import Network
import layers 
import manual_test


# load MNIST
(train_inputs, train_results), (test_inputs, test_results) = mnist.load_data()

# modify input shape from 6000, 28, 28 to 6000, 1, 28*28
train_inputs_new = train_inputs.reshape(train_inputs.shape[0], 1, 28*28)
test_inputs_new = test_inputs.reshape(test_inputs.shape[0], 1, 28*28)

#convert inputs to float type
train_inputs_new = train_inputs_new.astype('float32')
test_inputs_new = test_inputs_new.astype('float32')

#regularize between 0 and 1
train_inputs_new /= 255
test_inputs_new /= 255

# convert results to category array
train_results_new = np_utils.to_categorical(train_results)
test_results_new = np_utils.to_categorical(test_results)

# create the network
net = Network()
net.add(layers.FullyConnectedLayerWithActivation(28*28, 100, math_formulas.tanh, math_formulas.tanh_prime))
net.add(layers.FullyConnectedLayerWithActivation(100, 50, math_formulas.tanh, math_formulas.tanh_prime))
net.add(layers.FullyConnectedLayerWithActivation(50, 10, math_formulas.tanh, math_formulas.tanh_prime))

# hiperparameters
samples = 4000
epochs = 100
learning_rate = 0.1

# train the network
net.set_loss_function(math_formulas.mse, math_formulas.mse_prime)
net.train(train_inputs_new[0:samples], train_results_new[0:samples], epochs, learning_rate)

# predict test set
amount = 9
prediction = net.predict(test_inputs_new[0:amount])
res, pre = [], []
for i in range(amount):
    res.append(np.argmax(test_results_new[i]))
    pre.append(np.argmax(prediction[i]))

print("\ncorrect values:")
print(res)
print("predicted values:")
print(pre)

# predict from manual number
print("manual number prediction:")
# modify input shape from 1, 28, 28 to 1, 1, 28*28
test_input_manual_new = manual_test.test_input_manual.reshape(manual_test.test_input_manual.shape[0], 1, 28*28)
test_input_manual_ok_new = manual_test.test_input_manual_ok.reshape(manual_test.test_input_manual_ok.shape[0], 1, 28*28)

manual_prediction = net.predict(test_input_manual_new)
print(np.argmax(manual_prediction))

# plot MINST first 9 test images
for i in range(9):
    plt.subplot(4, 3, 1 + i)
    plt.imshow(test_inputs[i], cmap=plt.get_cmap('Blues'))

plt.subplot(4, 3, 11)
plt.imshow(manual_test.test_input_manual[0], cmap=plt.get_cmap('Blues'))

plt.show()

# plt.show()