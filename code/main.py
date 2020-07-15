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
from PIL import Image, ImageOps
from tensorflow.keras import layers, models, optimizers

print("Hello DL project!")

original_img_dir = 'data\\0_Images'
original_img_train_dir = join(original_img_dir, 'train')
original_img_test_dir = join(original_img_dir, 'test')
img_positions_dir = 'data\\5_DataDarkLines'

img_directories = [original_img_dir]

w = 4964
h = 7020




def fix_img_names():
    for directory in img_directories:
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        for idx, old_file_name in enumerate(files):
            file_name = '{}.jpg'.format(idx+1)
            old_file = join(directory, old_file_name)
            new_file = join(directory, file_name)
            os.rename(old_file, new_file)
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

#fix_img_names()
#exit()


'''
img_num = 2
line_info = load_img_line_info(img_num=img_num)
print('line info = ', line_info)

peaks_indices = line_info['peaks_indices'].flatten()
SCALE_FACTOR = line_info['SCALE_FACTOR'].flatten()[0]

y_positions = peaks_indices*SCALE_FACTOR
print('y_positions = ', y_positions)


top_test_area = line_info['top_test_area'].flatten()[0]
bottom_test_area = line_info['bottom_test_area'].flatten()[0]


img_path = join(original_img_dir, '{}.jpg'.format(img_num))
img = Image.open(img_path)
'''

'''
plt.imshow(img)

for y_pos in y_positions:
    plt.plot(np.arange(0, w), np.array([y_pos]*w))


#plt.plot(np.arange(0, w), np.array([top_test_area]*w))
#plt.plot(np.arange(0, w), np.array([bottom_test_area]*w))
plt.show()
'''

max_train_line_h = 200  # TODO:

'''
for idx in range(len(y_positions) - 1):
    if y_positions[idx + 1] < top_test_area or y_positions[idx] > bottom_test_area:
        sub_img = img.crop(box=(0, y_positions[idx], w, y_positions[idx+1]))
        file_name = '{0}_{1}.jpg'.format(img_num, idx+1)
        sub_img_file_path = original_img_train_dir + '/' + str(img_num) + '/' + file_name
        new_img = ImageOps.pad(sub_img, size=(w, max_train_line_h), color=(0xFF, 0xFF, 0xFF))
        new_img.save(sub_img_file_path)
        #plt.imshow(sub_img)
        #plt.show()


test_img = img.crop(box=(0, top_test_area, w, bottom_test_area))
#plt.imshow(test_img)
#plt.show()

test_img_file_path = join(original_img_test_dir, '{}.jpg'.format(img_num))
test_img.save(test_img_file_path)
'''


# create a data generator
train_gen = ImageDataGenerator()
train_dataset = train_gen.flow_from_directory(original_img_train_dir, class_mode='categorical', batch_size=2)

#num_of_writers = 204


def build_y_true():
    y_true = []
    files = [f for f in listdir(original_img_train_dir) if isfile(join(original_img_train_dir, f))]
    for file in files:
        last_pos = file.index('_')
        img_num = int(file[:last_pos])
        y_true.append(img_num)
    return y_true


#y_true = build_y_true()
#print('y_true=', y_true)
#n_of_cls = len(set(y_true))

'''
n_train_samples = num_of_writers // 2
n_validation_samples = 133
n_test_samples = num_of_writers - n_validation_samples - n_train_samples
'''
epochs = 10
batch_size = 2
learning_rate = 0.01

input_shape = (w, max_train_line_h, 3)
print('input_shape = ', input_shape)

model = models.Sequential()

model.add(layers.Dense(10, activation='relu', input_shape=input_shape))
model.add(layers.Dense(2, activation='softmax'))

opt = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, epochs=epochs, batch_size=batch_size)

