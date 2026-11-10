import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist # fashion image data set
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), # input shape of the images
                                # Layers in tf can have an activation fn run on them as seen in the parameters below.
                                    # relu function says if output < 0, set to 0
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    # softmax is often on the last layer; helps us find the most likely candidate out of the 10
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) # 10 neurons on the output size (10 possible clothing items)

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss, test_acc)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])