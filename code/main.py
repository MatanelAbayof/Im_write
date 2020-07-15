from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.io
import os
from os import listdir
from os.path import isfile, join
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

print("Hello DL project!")

original_img_dir = 'data\\0_Images'
original_img_test_dir = join(original_img_dir, 'test')
img_positions_dir = 'data\\5_DataDarkLines'

w = 4964
h = 7020

def fix_img_line_names():
    files = [f for f in listdir(img_positions_dir) if isfile(join(img_positions_dir, f))]
    for idx, old_file_name in enumerate(files):
        file_name = '{}.mat'.format(idx+1)
        old_file = join(img_positions_dir, old_file_name)
        new_file = join(img_positions_dir, file_name)
        os.rename(old_file, new_file)


def load_img_line_info(img_num: int):
    file_name = '{}.mat'.format(img_num)
    file_path = os.path.join(img_positions_dir, file_name)
    return scipy.io.loadmat(file_path)

img_num = 2
line_info = load_img_line_info(img_num=img_num)
print('line info = ', line_info)

top_test_area = line_info['top_test_area'].flatten()[0]
bottom_test_area = line_info['bottom_test_area'].flatten()[0]


img_path = join(original_img_dir, 'lines1_Page_02.jpg')
img = Image.open(img_path)
'''
plt.imshow(img)
plt.plot(np.arange(0, w), np.array([top_test_area]*w))
plt.plot(np.arange(0, w), np.array([bottom_test_area]*w))
plt.show()
'''

test_img = img.crop(box=(0, top_test_area, w, bottom_test_area))
plt.imshow(test_img)
plt.show()

test_img_file_path = join(original_img_test_dir, '{}.jpg'.format(img_num))
test_img.save(test_img_file_path)
#fix_img_line_names()



exit()
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

"""
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
"""
