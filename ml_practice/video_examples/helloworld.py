import tensorflow as tf
from tensorflow import keras
import numpy as np

"""
keras = the framework for NNs, sequential = a linear stack of layers
    - currently only one layer defined because the array in Sequential is of length 1
dense: every neuron in the layer is connected to every neuron in the previous layer. simplest neural layer type
"""
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error') # the optimiser updates the weights (makes a guess) and the loss function measures how wrong the guess is

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=1000)

print(model.predict(np.array([[10.0]], dtype=np.float32))) # have to use a numpy array because the model expects an array as input, not a single number