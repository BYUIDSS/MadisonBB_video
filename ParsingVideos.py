# Importing modules
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, VGG16
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout

# Saving videos as pictures
