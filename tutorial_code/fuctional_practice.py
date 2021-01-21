import tensorflow as tf

from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

# build functional model here

input = tf.keras.layers.Input((28, 28,))
x = tf.keras.layers.Flatten()(input)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x)

model = Model(inputs=[input], outputs=[output])

model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x=training_images, y=training_labels, epochs=9)
model.evaluate(test_images, test_labels)


