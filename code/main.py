from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

print("Hello DL project!")

# create a data generator
datagen = ImageDataGenerator()
dataset_dir = r'data\0_Images'
dataset = datagen.flow_from_directory(dataset_dir, class_mode='binary', batch_size=64)

num_of_writers = 407
y_true = np.arange(num_of_writers)

n_train_samples = 133
n_validation_samples = 133
n_test_samples = num_of_writers - n_validation_samples - n_train_samples
epochs = 10
batch_size = 5

img_width = 5000
img_height = 7000
input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=n_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=n_validation_samples // batch_size)
