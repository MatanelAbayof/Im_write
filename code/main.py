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

#region Global variables
CURRENT_DIR = os.path.dirname(__file__)
WORKING_DIR = Path(CURRENT_DIR).parent
os.chdir(WORKING_DIR)

IMG_ROOT_DIR = 'data'
ORIGINAL_IMG_DIR = join(IMG_ROOT_DIR, '0_Images')
ORIGINAL_IMG_TRAIN_DIR = join(ORIGINAL_IMG_DIR, 'train')
ORIGINAL_IMG_TEST_DIR = join(ORIGINAL_IMG_DIR, 'test')
IMG_POSITIONS_DIR = join(IMG_ROOT_DIR, '5_DataDarkLines')
IMG_DIRECTORIES = [ORIGINAL_IMG_DIR, IMG_POSITIONS_DIR]  # TODO: do this for each images directories

ORIGINAL_IMG_W = 4964
ORIGINAL_IMG_H = 7020
TRAIN_DATASET_RANGE = range(1, 204)
MAX_TRAIN_LINE_H = 205
MAX_TEST_LINE_H = 220
MAX_LINE_H = max(MAX_TRAIN_LINE_H, MAX_TEST_LINE_H)
#endregion

#region Functions
# -------------------------------------------------------------------------------------------------------------
def find_max_train_line_h():
    lines_h = []
    for img_num in TRAIN_DATASET_RANGE:
        print('checking image number ', img_num)
        line_info = load_img_lines_info(img_num)
        y_positions = parse_y_positions(line_info)
        for idx in range(len(y_positions) - 1):
            line_h = y_positions[idx+1] - y_positions[idx]
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
    # create a data generator
    train_gen = ImageDataGenerator()
    train_dataset = train_gen.flow_from_directory(ORIGINAL_IMG_TRAIN_DIR, class_mode='categorical', batch_size=2)

    # num_of_writers = 204

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

    model.fit(train_dataset, epochs=epochs, batch_size=batch_size) # don't need use deprecated function `fit_generator`
#endregion


is_find_max_train_line_h = False
is_find_max_test_line_h = False
is_fix_img_names = False
is_build_train_dataset = False
is_build_test_dataset = False
is_use_clf = True

print("Start project")
start_time = time.time()

#region Tasks
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
#endregion

end_time = time.time()
elapsed_time = end_time - start_time 
print('elapsed time = {:.3f} sec'.format(elapsed_time))
print('finish!')
