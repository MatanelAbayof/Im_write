import math
import os, sys
from os import listdir
from os.path import isfile, join
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
from tensorflow.keras import layers, models, optimizers
import shutil
import time

# region Global variables
CURRENT_DIR = os.path.dirname(__file__)
WORKING_DIR = Path(CURRENT_DIR).parent
os.chdir(WORKING_DIR)

IMG_ROOT_DIR = 'data'
ORIGINAL_IMG_DIR = join(IMG_ROOT_DIR, '0_Images')
ROTATED_IMG_DIR = join(IMG_ROOT_DIR, '1_ImagesRotated')
MEDIAN_IMG_DIR = join(IMG_ROOT_DIR, '2_ImagesMedianBW')
LINES_REMOVED_BW_IMG_DIR = join(IMG_ROOT_DIR, '3_ImagesLinesRemovedBW')
LINES_REMOVED_IMG_DIR = join(IMG_ROOT_DIR, '4_ImagesLinesRemoved')

ORIGINAL_IMG_TRAIN_DIR = join(ORIGINAL_IMG_DIR, 'train')
ORIGINAL_IMG_TEST_DIR = join(ORIGINAL_IMG_DIR, 'test')
IMG_POSITIONS_DIR = join(IMG_ROOT_DIR, '5_DataDarkLines')
IMG_DIRECTORIES = [ORIGINAL_IMG_DIR,
                   IMG_POSITIONS_DIR]  # TODO: do this for each images directories (IMG_POSITIONS_DIR must be the last item)

ORIGINAL_IMG_W = 4964
ORIGINAL_IMG_H = 7020
TRAIN_DATASET_RANGE = range(1, 3)  # TODO: change `stop` to 204
MAX_TRAIN_LINE_H = 205
MAX_TEST_LINE_H = 220
MAX_LINE_H = max(MAX_TRAIN_LINE_H, MAX_TEST_LINE_H)


# endregion

# region Functions
# -------------------------------------------------------------------------------------------------------------
def is_white_img(img: Image):  # TODO: fix this function
    inv_img = ImageOps.invert(img)
    print('inv_img = ', inv_img)
    white_box = inv_img.getbbox()
    print('bbox = ', white_box)
    return white_box


# -------------------------------------------------------------------------------------------------------------
def find_max_train_line_h():
    lines_h = []
    for img_num in TRAIN_DATASET_RANGE:
        print('checking image number ', img_num)
        line_info = load_img_lines_info(img_num)
        y_positions = parse_y_positions(line_info)
        for idx in range(len(y_positions) - 1):
            line_h = y_positions[idx + 1] - y_positions[idx]
            lines_h.append(line_h)
    print('lines height = ', lines_h)
    return max(lines_h)


# -------------------------------------------------------------------------------------------------------------
def find_max_test_line_h():
    lines_h = []
    for img_num in TRAIN_DATASET_RANGE:
        print('checking image number ', img_num)
        line_info = load_img_lines_info(img_num)
        top_test_area = line_info['top_test_area'].flatten()[0]
        bottom_test_area = line_info['bottom_test_area'].flatten()[0]
        line_h = bottom_test_area - top_test_area
        lines_h.append(line_h)
    print('lines height = ', lines_h)
    return max(lines_h)


# -------------------------------------------------------------------------------------------------------------
def fix_img_names():
    for dir_idx, directory in enumerate(IMG_DIRECTORIES):
        print('fixing directory "', directory, '"...')
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
    # removing directory
    shutil.rmtree(ORIGINAL_IMG_TRAIN_DIR)
    # create an empty directory
    Path(ORIGINAL_IMG_TRAIN_DIR).mkdir(parents=True, exist_ok=True)

    for img_num in TRAIN_DATASET_RANGE:
        build_train_dataset_img(img_num)


# -------------------------------------------------------------------------------------------------------------
def parse_y_positions(line_info):
    peaks_indices = line_info['peaks_indices'].flatten()
    scale_factor = line_info['SCALE_FACTOR'].flatten()[0]
    y_positions = peaks_indices * scale_factor
    return y_positions


# -------------------------------------------------------------------------------------------------------------
def build_train_dataset_img(img_num: int):
    print('building train dataset image num {}'.format(img_num))

    line_info = load_img_lines_info(img_num=img_num)
    y_positions = parse_y_positions(line_info)
    top_test_area = line_info['top_test_area'].flatten()[0]
    bottom_test_area = line_info['bottom_test_area'].flatten()[0]
    img = load_img(img_num)

    sub_img_dir = join(ORIGINAL_IMG_TRAIN_DIR, str(img_num))
    Path(sub_img_dir).mkdir(parents=True, exist_ok=True)

    for idx in range(len(y_positions) - 1):  # TODO: need also filter empty lines
        if y_positions[idx + 1] < top_test_area or y_positions[idx] > bottom_test_area:
            sub_img = img.crop(box=(0, y_positions[idx], ORIGINAL_IMG_W, y_positions[idx + 1]))
            file_name = '{0}_{1}.jpg'.format(img_num, idx + 1)
            sub_img_file_path = join(sub_img_dir, file_name)
            new_img = ImageOps.pad(sub_img, size=(ORIGINAL_IMG_W, MAX_LINE_H), color=(0xFF, 0xFF, 0xFF))
            new_img.save(sub_img_file_path)


# -------------------------------------------------------------------------------------------------------------
def build_all_test_dataset():
    # removing directory
    shutil.rmtree(ORIGINAL_IMG_TEST_DIR)
    # create an empty directory
    Path(ORIGINAL_IMG_TEST_DIR).mkdir(parents=True, exist_ok=True)

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


# -------------------------------------------------------------------------------------------------------------
def use_clf():
    img_scale = 0.5
    target_size = (math.floor(float(ORIGINAL_IMG_W)*img_scale), math.floor(float(MAX_LINE_H)*img_scale))
    print('target_size = ', target_size)
    # create a data generator
    train_gen = ImageDataGenerator()  # TODO: use `rotation_range` &  `width_shift_range` & `height_shift_range`
    train_dataset = train_gen.flow_from_directory(ORIGINAL_IMG_TRAIN_DIR, target_size=target_size,
                                                  class_mode='categorical', batch_size=2)

    num_of_cls = 5

    '''
    n_train_samples = num_of_writers // 2
    n_validation_samples = 133
    n_test_samples = num_of_writers - n_validation_samples - n_train_samples
    '''
    epochs = 2
    batch_size = 2
    learning_rate = 0.01

    input_shape = (*target_size, 3)
    print('input_shape = ', input_shape)

    model = models.Sequential()
    '''
    model.add(layers.Dense(32, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(num_of_writers, activation='softmax'))  # TODO: `units` must be number of classes
    '''

    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_of_cls, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=learning_rate),
                  metrics=['acc'])

    # TODO: use `validation_data` & `validation_steps`
    model.fit(train_dataset, epochs=epochs, batch_size=batch_size)  # don't need use deprecated function `fit_generator`


# endregion

'''
img_path = join(ORIGINAL_IMG_TRAIN_DIR, '1/1_6.jpg')
img = Image.open(img_path)
res = is_white_img(img)
print('res = ', res)
exit()
'''

is_find_max_train_line_h = False
is_find_max_test_line_h = False
is_fix_img_names = False
is_build_train_dataset = False
is_build_test_dataset = False
is_use_clf = True

print("Start project")
start_time = time.time()

# region Tasks
if is_find_max_train_line_h:
    print('find maximum train line height...')
    max_train_line_h = find_max_train_line_h()
    print('maximum train line height = ', max_train_line_h)

if is_find_max_test_line_h:
    print('find maximum test line height...')
    max_test_line_h = find_max_test_line_h()
    print('maximum test line height = ', max_test_line_h)

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

if is_use_clf:
    print('using classifier...')
    use_clf()
    print('finish use classifier successfully')
# endregion

end_time = time.time()
elapsed_time = end_time - start_time
print('elapsed time = {:.3f} sec'.format(elapsed_time))
print('finish!')
