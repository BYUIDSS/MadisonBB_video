# %%
# Load Libraries
import tensorflow as tf
import pandas as pd
import numpy as np

# %%
# Load data
mnist = tf.keras.datasets.mnist

# Make train and test Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 ### Greyscale images go from a scale of 0 to 255. This line standardizes the data (0 to 1)

# %%
# build the model

model = tf.keras.models.Sequential([ # Sequential means 1 by 1 in order
  tf.keras.layers.Flatten(input_shape=(28, 28)), # Flatten data into a vector with one column to feed into the first layer of the neural network
  tf.keras.layers.Dense(128, activation='relu'), # 128 neurons within the layer
  tf.keras.layers.Dropout(0.2), # Makes this less complex by deleting 20% of the neurons to generalize better (Don't want the model to be to complex)
  tf.keras.layers.Dense(10) # Make another dense layer of 10 neurons (Digit 0 through digit 9). """This is the output layer.""""
])

predictions = model(x_train[:1]).numpy() 
print(predictions)

