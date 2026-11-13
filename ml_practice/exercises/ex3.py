"""
tf.keras.layers.Conv2D(
    filters, = number of filters
    kernel_size, = filter size. int 3 or tuple (3, 3) for height/width
    strides=(1, 1), = how far window moves at each step
    padding="valid", = no padding or padding
    activation=None, = activation fn to apply to output
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    **kwargs
)

Notes (Theory):
- Convolutions use filters/kernels to narrow down the content of the image to focus on distinct details
- Convolutional layers are used for image data, and they are often followed by pooling layers
- In a nutshell, 'overfitting' occurs when the network learns the data from the training set really well, 
  but it's too specialised to only that data, and as a result is less effective at seeing other data.
  e.g. if you only saw red shoes for your entire life, you may be confused looking at blue shoes
- Typical architecture of a CNN:
    - Convolutional layer (applies filters = matrices which perform element-wise multiplication)
    - ReLU (rectified linear unit = activation function, introduces non linearity by turning neg vals to zero)
    - Pooling (reduces size while keeping important features)
    - Dense (fully connected) layer (converts 2D into 1D vector; often using softmax for probabilities)
- A tensor is a mathematical object that generalises scalars, vectors, and matrices to an arbitrary number of dimensions

Notes (Practice):
- can use model.summary() to see how data is changing as it goes through the network

Exercise: 

In the videos you looked at how you would improve Fashion MNIST using Convolutions. For your exercise see if you can improve MNIST 
to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. You should stop training once the accuracy
goes above this amount. It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training, but your 
training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.

I've started the code for you -- you need to finish it!

When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"
"""

import tensorflow as tf

# Define callback conditions
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.998):
      print("\nReached 99.8% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# We need to reshape the images into a 4D list that is 60,000x28x28x1, because the first convolution (Conv2D) expects a 4D tensor
# So we specify the input shape to be 28 * 28 * 1
# The 1 represents greyscale, this would be 3 instead for RGB
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

callback = myCallback()

# Configure how the model will learn
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using training data
model.fit(training_images, training_labels, epochs=10, callbacks=[callback])

# Test performance on unseen data
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Display test accuracy (percentage)
print(test_acc)
# YOUR CODE ENDS HERE
