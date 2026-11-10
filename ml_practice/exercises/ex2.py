"""
In the course you learned how to do classification using Fashion MNIST, a data set containing items of clothing.
There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs 
-- i.e. you should stop training once you reach that level of accuracy.

Some notes:
- It should succeed in less than 10 epochs, so it is okay to change epochs to 10, but nothing larger
- When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
- If you add any additional variables, make sure you use the same names as the ones used in the class
"""

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist # handwritten digit data set

# load data and verify the original image representations to normalise
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
print(training_images.dtype)  # uint8
print(training_images.min())  # 0
print(training_images.max())  # 255

training_images, test_images = training_images / 255.0 , test_images / 255.0 # normalise

# save training image example to file
output_path = Path(__file__).with_name("ex2_output.png")
plt.imsave(output_path, training_images[0], cmap="gray")

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), # input shape of the images
                                # Layers in tf can have an activation fn run on them as seen in the parameters below.
                                    # relu function says if output < 0, set to 0
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
                                    # softmax is often on the last layer; helps us find the most likely candidate out of the 10
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) # 10 neurons on the output size (10 possible clothing items)

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(test_loss, test_acc)

