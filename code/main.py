from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

print("Hello DL project!")

# create a data generator
datagen = ImageDataGenerator()
dataset_dir = r'data\0_Images'
dataset = datagen.flow_from_directory(dataset_dir, class_mode='binary', batch_size=64)

num_of_writers = 407
y_true = np.arange(num_of_writers)