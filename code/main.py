import math
import os, sys
import pytesseract
from pytesseract import Output
import cv2
from os import listdir
from os.path import isfile, join, isdir
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
from keras.optimizers import SGD
#from convnetskeras.convnets import preprocess_image_batch
#from convnetskeras.convnets import convnet
from alex import AlexNet

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
IMG_POSITIONS_DIR = join(IMG_ROOT_DIR, '5_DataDarkLines')
ALEX_WEIGHTS_PATH = join(IMG_ROOT_DIR, 'alexnet_weights.h5')
IMG_DIRECTORIES = [ORIGINAL_IMG_DIR, LINES_REMOVED_BW_IMG_DIR, IMG_POSITIONS_DIR]  # TODO: do this for each images directories (IMG_POSITIONS_DIR must be the last item)

ORIGINAL_IMG_W = 4964
ORIGINAL_IMG_H = 7020
TRAIN_DATASET_RANGE = range(1, 3)  # TODO: change `stop` to 204
MAX_TRAIN_LINE_H = 205
MAX_TEST_LINE_H = 220
MAX_LINE_H = max(MAX_TRAIN_LINE_H, MAX_TEST_LINE_H)
MAX_WORD_W = 776
MAX_WORD_H = 194

MIN_WORD_W = 40
MIN_WORD_H = 40


# endregion

# region Functions
# -------------------------------------------------------------------------------------------------------------
def is_white_img(img: Image): # TODO: improve
    imageSizeW, imageSizeH = img.size
    nonWhitePixels = []  
    for i in range(1, imageSizeW):
        for j in range(1, imageSizeH):
            pixVal = img.getpixel((i, j))
            if pixVal != (255, 255, 255):
                nonWhitePixels.append(pixVal)
    return len(nonWhitePixels) < 2 * imageSizeW


# -------------------------------------------------------------------------------------------------------------
def find_max_train_line_h():
    lines_h = []
    for img_num in TRAIN_DATASET_RANGE:
        print('checking image number ', img_num)
        line_info = load_img_lines_info(img_num=img_num)
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
def load_img(dir: str = ORIGINAL_IMG_DIR, img_num: int = 1):
    img_path = join(dir, '{}.jpg'.format(img_num))
    return Image.open(img_path)


# -------------------------------------------------------------------------------------------------------------
def load_img_lines_info(img_num: int):
    file_name = '{}.mat'.format(img_num)
    file_path = os.path.join(IMG_POSITIONS_DIR, file_name)
    return scipy.io.loadmat(file_path)


# -------------------------------------------------------------------------------------------------------------
def build_all_train_dataset(img_dir: str):
    train_dir = get_train_dir(img_dir)
    # removing directory
    shutil.rmtree(train_dir, ignore_errors=True)
    # create an empty directory
    Path(train_dir).mkdir(parents=True, exist_ok=True)

    for img_num in TRAIN_DATASET_RANGE:
        build_train_dataset_img(img_dir=img_dir, img_num=img_num)


# -------------------------------------------------------------------------------------------------------------
def parse_y_positions(line_info):
    peaks_indices = line_info['peaks_indices'].flatten()
    scale_factor = line_info['SCALE_FACTOR'].flatten()[0]
    y_positions = peaks_indices * scale_factor
    return y_positions


# -------------------------------------------------------------------------------------------------------------
def get_train_dir(img_dir: str):
    return join(img_dir, 'train')

# -------------------------------------------------------------------------------------------------------------
def get_test_dir(img_dir: str):
    return join(img_dir, 'test')

# -------------------------------------------------------------------------------------------------------------
def build_train_dataset_img(img_dir: str, img_num: int):
    print('building train dataset image num {}'.format(img_num))

    train_dir = get_train_dir(img_dir)

    line_info = load_img_lines_info(img_num=img_num)
    y_positions = parse_y_positions(line_info)
    top_test_area = line_info['top_test_area'].flatten()[0]
    bottom_test_area = line_info['bottom_test_area'].flatten()[0]
    img = load_img(dir=img_dir, img_num=img_num)

    sub_img_dir = join(train_dir, str(img_num))
    Path(sub_img_dir).mkdir(parents=True, exist_ok=True)

    for idx in range(len(y_positions) - 1):
        if y_positions[idx + 1] < top_test_area or y_positions[idx] > bottom_test_area:
            sub_img = img.crop(box=(0, y_positions[idx], ORIGINAL_IMG_W, y_positions[idx + 1]))
            file_name = '{0}_{1}.jpg'.format(img_num, idx + 1)
            sub_img_file_path = join(sub_img_dir, file_name)
            new_img = ImageOps.pad(sub_img.convert("RGB"), size=(ORIGINAL_IMG_W, MAX_LINE_H), color=(0xFF, 0xFF, 0xFF))
            if not is_white_img(new_img):
                new_img.save(sub_img_file_path)


# -------------------------------------------------------------------------------------------------------------
def build_all_test_dataset(img_dir: str):
    test_dir = get_test_dir(img_dir)
    # removing directory
    shutil.rmtree(test_dir, ignore_errors=True)
    # create an empty directory
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    for img_num in TRAIN_DATASET_RANGE:
        build_test_dataset_img(img_dir=img_dir, img_num=img_num)


# -------------------------------------------------------------------------------------------------------------
def build_all_train_words_dataset(img_dir: str):
    for img_num in TRAIN_DATASET_RANGE:
        build_train_words_dataset_img(imgs_dir=img_dir, img_num=img_num)

# -------------------------------------------------------------------------------------------------------------
def build_train_words_dataset_img(imgs_dir: str, img_num: int):
    print('building image words for image num {}...'.format(img_num))
    imgs_train_dir = get_train_dir(imgs_dir)
    img_dir_path = join(imgs_train_dir, str(img_num))
    lines_files_names = [f for f in listdir(img_dir_path) if isfile(join(img_dir_path, f))]
    

    img_words_path = join(img_dir_path, 'words')
    shutil.rmtree(img_words_path, ignore_errors=True)
    Path(img_words_path).mkdir(parents=True, exist_ok=True) 

    for line_file_name in lines_files_names:
        start_idx_line_n = line_file_name.index('_') + 1
        end_idx_line_n = line_file_name.index('.')
        row_num = line_file_name[start_idx_line_n:end_idx_line_n]
        img_line_path = join(img_dir_path, '{0}_{1}.jpg'.format(img_num, row_num))
        img_line = Image.open(img_line_path)
        #img = cv2.imread(img_path)
        img_line_data = pytesseract.image_to_data(img_line, output_type=Output.DICT)
        n_boxes = len(img_line_data['word_num'])
        for i in range(n_boxes):
            if img_line_data['word_num'][i] > 0:
                (x, y, w, h) = (img_line_data['left'][i], img_line_data['top'][i], img_line_data['width'][i], img_line_data['height'][i])
                #new_img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img_word = img_line.crop(box= (x, y, x+w, y+h))
                img_w, img_h = img_word.size
                if (img_w >= MIN_WORD_W and img_h >= MIN_WORD_H) and not is_white_img(img_word):
                    img_word_path = join(img_words_path, '{0}_{1}_{2}.jpg'.format(img_num, row_num, i))
                    img_word.save(img_word_path)

# -------------------------------------------------------------------------------------------------------------
def for_each_img_word(dadaset_dir: str):
    imgs_dirs_names = [d for d in listdir(dadaset_dir) if isdir(join(dadaset_dir, d))]
    for img_dir_name in imgs_dirs_names:
        print('scanning image number {}...'.format(img_dir_name))
        img_dir = join(dadaset_dir, img_dir_name)
        words_dir = join(img_dir, 'words')
        if not os.path.exists(words_dir):
            break
        imgs_words_names = [f for f in listdir(words_dir) if isfile(join(words_dir, f))]
        for img_word_name in imgs_words_names:
            img_word_file_path = join(words_dir, img_word_name)
            img_word = Image.open(img_word_file_path)
            yield img_word_file_path, img_word


# -------------------------------------------------------------------------------------------------------------
def find_max_word_size(imgs_dir: str):
    max_w, max_h = (0, 0)
    train_dir = get_train_dir(imgs_dir)
    for _, img_word in for_each_img_word(train_dir):
            img_w, img_h = img_word.size
            max_w = max(img_w, max_w)
            max_h = max(img_h, max_h)
    return max_w, max_h
# -------------------------------------------------------------------------------------------------------------
def build_test_dataset_img(img_dir: str, img_num: int):
    print('building test dataset image num {}'.format(img_num))

    test_dir = get_test_dir(img_dir)

    line_info = load_img_lines_info(img_num=img_num)
    top_test_area = line_info['top_test_area'].flatten()[0]
    bottom_test_area = line_info['bottom_test_area'].flatten()[0]
    img = load_img(dir=img_dir, img_num=img_num)
    test_img = img.crop(box=(0, top_test_area, ORIGINAL_IMG_W, bottom_test_area))
    pad_test_img = ImageOps.pad(test_img.convert("RGB"), size=(ORIGINAL_IMG_W, MAX_LINE_H), color=(0xFF, 0xFF, 0xFF))
    test_img_dir_path = join(test_dir, str(img_num))
    Path(test_img_dir_path).mkdir(parents=True, exist_ok=True)
    test_img_file_path = join(test_img_dir_path, '{}.jpg'.format(img_num))  # TODO: save in nested folder `img_name`
    if not is_white_img(pad_test_img):
        pad_test_img.save(test_img_file_path)


# -------------------------------------------------------------------------------------------------------------
def use_clf(img_dir: str):
    train_dir = get_train_dir(img_dir)

    img_scale = 1.
    target_size = (math.floor(float(ORIGINAL_IMG_W)*img_scale), math.floor(float(MAX_LINE_H)*img_scale))
    print('target_size = ', target_size)

    # create a data generator
    shift_side = 0.1
    train_gen = ImageDataGenerator(rotation_range=2, width_shift_range=shift_side, height_shift_range=shift_side)
    train_dataset = train_gen.flow_from_directory(train_dir, target_size=target_size,
                                                  class_mode='categorical', batch_size=1)
    #TODO: add validation_gen, test_gen

    num_of_cls = len(train_dataset.class_indices)
    sample_count = len(train_dataset.filenames)


    '''
    n_train_samples = num_of_writers // 2
    n_validation_samples = 133
    n_test_samples = num_of_writers - n_validation_samples - n_train_samples
    '''
    epochs = 2
    batch_size = 1
    learning_rate = 0.05
    steps_per_epoch = sample_count // num_of_cls

    input_shape = (*target_size, 3)
    print('input_shape = ', input_shape)

    model = models.Sequential()
    #model = AlexNet(weights_path=ALEX_WEIGHTS_PATH, heatmap=False)

    model.add(layers.Conv2D(8, (24, 12), activation='relu',
                            input_shape=input_shape))
    model.add(layers.MaxPooling2D((4, 4)))
    model.add(layers.Conv2D(8, (6, 6), activation='relu'))
    model.add(layers.MaxPooling2D((4, 4)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_of_cls, activation='softmax'))


    model.summary()


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=learning_rate),
                  metrics=['acc'])

    # TODO: use `validation_data` & `validation_steps`
    model.fit(train_dataset, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch)  # don't need use deprecated function `fit_generator`

# -------------------------------------------------------------------------------------------------------------
def pad_imgs_words(imgs_dir: str):
    train_dir = get_train_dir(imgs_dir)
    for img_word_file_path, img_word in for_each_img_word(train_dir):
        new_img = ImageOps.pad(img_word.convert("RGB"), size=(MAX_WORD_W, MAX_WORD_H), color=(0xFF, 0xFF, 0xFF))
        img_word.close()
        new_img.save(img_word_file_path)


# endregion


is_find_max_train_line_h = False
is_find_max_test_line_h = False
is_fix_img_names = False
is_build_train_dataset = False
is_build_test_dataset = False
is_use_clf = True
is_build_all_train_words_dataset = False
is_find_max_word_size = False
is_pad_imgs_words = False

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
    build_img_dir = LINES_REMOVED_BW_IMG_DIR
    build_all_train_dataset(build_img_dir)
    print('train dataset has built successfully')

if is_build_test_dataset:
    print('building test dataset...')
    build_img_dir = LINES_REMOVED_BW_IMG_DIR
    build_all_test_dataset(build_img_dir)
    print('test dataset has built successfully')

if is_build_all_train_words_dataset:
    print('building words of images dataset...')
    build_img_dir = LINES_REMOVED_BW_IMG_DIR
    build_all_train_words_dataset(build_img_dir)
    print('train images words dataset has built successfully')

if is_find_max_word_size:
    print('find maximum word size...')
    img_dir = LINES_REMOVED_BW_IMG_DIR
    max_w, max_h = find_max_word_size(img_dir)
    print('maximum width is', max_w)
    print('maximum height is', max_h)
    print('maximum word size has found successfully')

if is_pad_imgs_words:
    print('padding images words...')
    imgs_dir = LINES_REMOVED_BW_IMG_DIR
    pad_imgs_words(imgs_dir)
    print('finish pad images words successfully')

if is_use_clf:
    print('using classifier...')
    img_dir = LINES_REMOVED_BW_IMG_DIR
    use_clf(img_dir)
    print('finish use classifier successfully')
# endregion

end_time = time.time()
elapsed_time = end_time - start_time
print('elapsed time = {:.3f} sec'.format(elapsed_time))
print('finish!')
