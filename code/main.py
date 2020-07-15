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
from pathlib import Path

IMG_ROOT_DIR = 'data'
ORIGINAL_IMG_DIR = join(IMG_ROOT_DIR, '0_Images')
ORIGINAL_IMG_TRAIN_DIR = join(ORIGINAL_IMG_DIR, 'train')
ORIGINAL_IMG_TEST_DIR = join(ORIGINAL_IMG_DIR, 'test')
IMG_POSITIONS_DIR = join(IMG_ROOT_DIR, '5_DataDarkLines')

IMG_DIRECTORIES = [ORIGINAL_IMG_DIR, IMG_POSITIONS_DIR]  # TODO: do this for each images directories

ORIGINAL_IMG_W = 4964
ORIGINAL_IMG_H = 7020
TRAIN_DATASET_RANGE = range(1, 204)


# -------------------------------------------------------------------------------------------------------------
def fix_img_names():
    for dir_idx, directory in enumerate(IMG_DIRECTORIES):
        file_ext = 'mat' if (dir_idx == len(IMG_DIRECTORIES) - 1) else 'jpg'
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        for idx, old_file_name in enumerate(files):
            file_name = '{0}.{1}'.format(idx + 1, file_ext)
            old_file = join(directory, old_file_name)
            new_file = join(directory, file_name)
            os.rename(old_file, new_file)


# -------------------------------------------------------------------------------------------------------------
def load_img(img_num: int):
    img_path = join(ORIGINAL_IMG_DIR, '{}.jpg'.format(img_num))
    return Image.open(img_path)


# -------------------------------------------------------------------------------------------------------------
def load_img_lines_info(img_num: int):
    file_name = '{}.mat'.format(img_num)
    file_path = os.path.join(IMG_POSITIONS_DIR, file_name)
    return scipy.io.loadmat(file_path)


# -------------------------------------------------------------------------------------------------------------
def build_all_train_dataset():
    for img_num in TRAIN_DATASET_RANGE:
        build_train_dataset_img(img_num)


# -------------------------------------------------------------------------------------------------------------
def build_train_dataset_img(img_num: int):
    print('building train dataset image num {}'.format(img_num))

    line_info = load_img_lines_info(img_num=img_num)
    peaks_indices = line_info['peaks_indices'].flatten()
    scale_factor = line_info['SCALE_FACTOR'].flatten()[0]
    y_positions = peaks_indices * scale_factor
    top_test_area = line_info['top_test_area'].flatten()[0]
    bottom_test_area = line_info['bottom_test_area'].flatten()[0]
    img = load_img(img_num)

    max_train_line_h = 200  # TODO: load maximum line height at all images files

    sub_img_dir = join(ORIGINAL_IMG_TRAIN_DIR, str(img_num))
    Path(sub_img_dir).mkdir(parents=True, exist_ok=True)

    for idx in range(len(y_positions) - 1):  # TODO: need also filter empty lines
        if y_positions[idx + 1] < top_test_area or y_positions[idx] > bottom_test_area:
            sub_img = img.crop(box=(0, y_positions[idx], ORIGINAL_IMG_W, y_positions[idx + 1]))
            file_name = '{0}_{1}.jpg'.format(img_num, idx + 1)
            sub_img_file_path = join(sub_img_dir, file_name)
            new_img = ImageOps.pad(sub_img, size=(ORIGINAL_IMG_W, max_train_line_h), color=(0xFF, 0xFF, 0xFF))
            new_img.save(sub_img_file_path)


# -------------------------------------------------------------------------------------------------------------
def build_all_test_dataset():
    for img_num in TRAIN_DATASET_RANGE:
        build_test_dataset_img(img_num)


# -------------------------------------------------------------------------------------------------------------
def build_test_dataset_img(img_num: int):
    print('building test dataset image num {}'.format(img_num))

    line_info = load_img_lines_info(img_num=img_num)
    top_test_area = line_info['top_test_area'].flatten()[0]
    bottom_test_area = line_info['bottom_test_area'].flatten()[0]
    img = load_img(img_num)
    test_img = img.crop(box=(0, top_test_area, ORIGINAL_IMG_W, bottom_test_area))
    # TODO: need to pad image with `max_test_line_h`
    test_img_file_path = join(ORIGINAL_IMG_TEST_DIR, '{}.jpg'.format(img_num))  # TODO: save in nested folder `img_name`
    test_img.save(test_img_file_path)


is_fix_img_names = False
is_build_train_dataset = False
is_build_test_dataset = False

print("Start project")

if is_fix_img_names:
    print('fixing images names...')
    fix_img_names()
    print('finish fix images names successfully')

if is_build_train_dataset:
    print('building train dataset...')
    build_all_train_dataset()
    print('train dataset has built successfully')

if is_build_test_dataset:
    print('building test dataset...')
    build_all_train_dataset()
    print('test dataset has built successfully')

exit()

# create a data generator
train_gen = ImageDataGenerator()
train_dataset = train_gen.flow_from_directory(ORIGINAL_IMG_TRAIN_DIR, class_mode='categorical', batch_size=2)


# num_of_writers = 204


def build_y_true():
    y_true = []
    files = [f for f in listdir(ORIGINAL_IMG_TRAIN_DIR) if isfile(join(ORIGINAL_IMG_TRAIN_DIR, f))]
    for file in files:
        last_pos = file.index('_')
        img_num = int(file[:last_pos])
        y_true.append(img_num)
    return y_true


# y_true = build_y_true()
# print('y_true=', y_true)
# n_of_cls = len(set(y_true))

'''
n_train_samples = num_of_writers // 2
n_validation_samples = 133
n_test_samples = num_of_writers - n_validation_samples - n_train_samples
'''
epochs = 10
batch_size = 2
learning_rate = 0.01

input_shape = (ORIGINAL_IMG_W, max_train_line_h, 3)
print('input_shape = ', input_shape)

model = models.Sequential()

model.add(layers.Dense(10, activation='relu', input_shape=input_shape))
model.add(layers.Dense(2, activation='softmax'))  # TODO: `units` must be number of classes

opt = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, epochs=epochs, batch_size=batch_size)
