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
from keras.applications import VGG16, ResNet50
import ntpath

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# region Global variables
CURRENT_DIR = os.path.dirname(__file__)
WORKING_DIR = Path(CURRENT_DIR).parent
os.chdir(WORKING_DIR)

TRAIN_FEATURES_FILE_NAME = "train_features"
TRAIN_LABELS_FILE_NAME = "train_labels"

IMG_ROOT_DIR = 'data'
ORIGINAL_IMG_DIR = join(IMG_ROOT_DIR, '0_Images')
ROTATED_IMG_DIR = join(IMG_ROOT_DIR, '1_ImagesRotated')
MEDIAN_IMG_DIR = join(IMG_ROOT_DIR, '2_ImagesMedianBW')
LINES_REMOVED_BW_IMG_DIR = join(IMG_ROOT_DIR, '3_ImagesLinesRemovedBW')
LINES_REMOVED_IMG_DIR = join(IMG_ROOT_DIR, '4_ImagesLinesRemoved')
IMG_POSITIONS_DIR = join(IMG_ROOT_DIR, '5_DataDarkLines')
ALEX_WEIGHTS_PATH = join(IMG_ROOT_DIR, 'alexnet_weights.h5')
IMG_DIRECTORIES = [ORIGINAL_IMG_DIR, LINES_REMOVED_BW_IMG_DIR,
                   IMG_POSITIONS_DIR]  # TODO: do this for each images directories (IMG_POSITIONS_DIR must be the last item)

ORIGINAL_IMG_W = 4964
ORIGINAL_IMG_H = 7020
TRAIN_DATASET_RANGE = range(1, 3)  # TODO: change `stop` to 204
MAX_TRAIN_LINE_H = 205
MAX_TEST_LINE_H = 220
MAX_LINE_H = max(MAX_TRAIN_LINE_H, MAX_TEST_LINE_H)
MAX_WORD_W = 776
MAX_WORD_H = 194
MAX_WORD_SIZE = (MAX_WORD_W, MAX_WORD_H)
MIN_WORD_W = 40
MIN_WORD_H = 40


# endregion

# region Functions
# -------------------------------------------------------------------------------------------------------------
def full_build_dataset(imgs_dir: str):
    '''
    print('fixing images names...')
    fix_imgs_names()
    print('finding maximum train line height...')
    max_train_line_h = find_max_train_line_h()
    print('max_train_line_h =', max_train_line_h)
    print('finding maximum test line height...')
    max_test_line_h = find_max_test_line_h()
    print('max_test_line_h =', max_test_line_h)
    max_line_h = max(max_train_line_h, max_test_line_h)
    print('max_line_h =', max_line_h)
    print('building lines at train dataset...')
    build_all_lines_train_dataset(imgs_dir, max_line_h=max_line_h)
    print('images lines at train dataset has built successfully')
    print('building lines at test dataset...')
    build_all_lines_test_dataset(imgs_dir, max_line_h=max_line_h)
    print('images lines at test dataset has built successfully')   
    print('building words of images at train dataset...')
    build_all_words_dataset(train_dir)
    print('train images words dataset has built successfully')
    print('building words of images at test dataset...')
    '''
    build_all_words_dataset(test_dir) #TODO: fix indexof here
    print('test images words dataset has built successfully')
    print('find maximum word size...')
    max_w, max_h = find_max_word_size(imgs_dir)
    max_word_size = (max_w, max_h)
    print('maximum word size is', max_word_size)
    pad_imgs_words(imgs_dir=imgs_dir, max_word_size=max_word_size)
    #TODO: use move words to dataset folder and split to validation folder

    
# -------------------------------------------------------------------------------------------------------------
def get_train_dir(imgs_dir: str):
    return join(imgs_dir, 'train')

# -------------------------------------------------------------------------------------------------------------
def get_validation_dir(imgs_dir: str):
    return join(imgs_dir, 'validation')

# -------------------------------------------------------------------------------------------------------------
def get_test_dir(imgs_dir: str):
    return join(imgs_dir, 'test')

# -------------------------------------------------------------------------------------------------------------
def is_white_img(img: Image):  # TODO: improve run time
    img_w, img_h = img.size
    not_white_pixels = []
    for i in range(1, img_w):
        for j in range(1, img_h):
            pixel = img.getpixel((i, j))
            if pixel != (255, 255, 255):
                not_white_pixels.append(pixel)
    return len(not_white_pixels) < 2 * img_w

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
def fix_imgs_names():
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
def parse_y_positions(line_info):
    peaks_indices = line_info['peaks_indices'].flatten()
    scale_factor = line_info['SCALE_FACTOR'].flatten()[0]
    y_positions = peaks_indices * scale_factor
    return y_positions

# -------------------------------------------------------------------------------------------------------------
def build_all_lines_train_dataset(img_dir: str, max_line_h = MAX_LINE_H):
    train_dir = get_train_dir(img_dir)
    # removing directory
    shutil.rmtree(train_dir, ignore_errors=True)
    # create an empty directory
    Path(train_dir).mkdir(parents=True, exist_ok=True)

    for img_num in TRAIN_DATASET_RANGE:
        build_lines_train_dataset_img(imgs_dir=img_dir, img_num=img_num, max_line_h=max_line_h)

# -------------------------------------------------------------------------------------------------------------
def build_lines_train_dataset_img(imgs_dir: str, img_num: int, max_line_h = MAX_LINE_H):
    print('building lines of train dataset image num {}'.format(img_num))

    train_dir = get_train_dir(imgs_dir)

    line_info = load_img_lines_info(img_num=img_num)
    y_positions = parse_y_positions(line_info)
    top_test_area = line_info['top_test_area'].flatten()[0]
    bottom_test_area = line_info['bottom_test_area'].flatten()[0]
    img = load_img(dir=imgs_dir, img_num=img_num)

    sub_img_dir = join(train_dir, str(img_num))
    Path(sub_img_dir).mkdir(parents=True, exist_ok=True)

    for idx in range(len(y_positions) - 1):
        if y_positions[idx + 1] < top_test_area or y_positions[idx] > bottom_test_area:
            sub_img = img.crop(box=(0, y_positions[idx], ORIGINAL_IMG_W, y_positions[idx + 1]))
            file_name = '{0}_{1}.jpg'.format(img_num, idx + 1)
            sub_img_file_path = join(sub_img_dir, file_name)
            new_img = ImageOps.pad(sub_img.convert("RGB"), size=(ORIGINAL_IMG_W, max_line_h), color=(0xFF, 0xFF, 0xFF))
            if not is_white_img(new_img):
                new_img.save(sub_img_file_path)

# -------------------------------------------------------------------------------------------------------------
def build_all_lines_test_dataset(imgs_dir: str, max_line_h = MAX_LINE_H):
    test_dir = get_test_dir(imgs_dir)
    # removing directory
    shutil.rmtree(test_dir, ignore_errors=True)
    # create an empty directory
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    for img_num in TRAIN_DATASET_RANGE:
        build_lines_test_dataset_img(img_dir=imgs_dir, img_num=img_num, max_line_h=max_line_h)

# -------------------------------------------------------------------------------------------------------------
def build_lines_test_dataset_img(img_dir: str, img_num: int, max_line_h = MAX_LINE_H):
    print('building test dataset image num {}'.format(img_num))

    test_dir = get_test_dir(img_dir)

    line_info = load_img_lines_info(img_num=img_num)
    top_test_area = line_info['top_test_area'].flatten()[0]
    bottom_test_area = line_info['bottom_test_area'].flatten()[0]
    img = load_img(dir=img_dir, img_num=img_num)
    test_img = img.crop(box=(0, top_test_area, ORIGINAL_IMG_W, bottom_test_area))
    pad_test_img = ImageOps.pad(test_img.convert("RGB"), size=(ORIGINAL_IMG_W, max_line_h), color=(0xFF, 0xFF, 0xFF))
    test_img_dir_path = join(test_dir, str(img_num))
    Path(test_img_dir_path).mkdir(parents=True, exist_ok=True)
    test_img_file_path = join(test_img_dir_path, '{}.jpg'.format(img_num))  # TODO: save in nested folder `img_name`
    pad_test_img.save(test_img_file_path)

# -------------------------------------------------------------------------------------------------------------
def build_all_words_dataset(dataset_dir: str):
    for img_num in TRAIN_DATASET_RANGE:
       build_words_dataset_img(dataset_dir=dataset_dir, img_num=img_num)

# -------------------------------------------------------------------------------------------------------------
def build_words_dataset_img(dataset_dir: str, img_num: int):
    print('building image words for image num {}...'.format(img_num))
    img_dir_path = join(dataset_dir, str(img_num))
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
        # img = cv2.imread(img_path)
        img_line_data = pytesseract.image_to_data(img_line, output_type=Output.DICT)
        n_boxes = len(img_line_data['word_num'])
        for i in range(n_boxes):
            if img_line_data['word_num'][i] > 0:
                (x, y, w, h) = (img_line_data['left'][i], img_line_data['top'][i], img_line_data['width'][i],
                                img_line_data['height'][i])
                # new_img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img_word = img_line.crop(box=(x, y, x + w, y + h))
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
def extract_features(directory, sample_count, dataset, batch_size, input_shape, target_size):
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    conv_base.summary()

    conv_base_output_shape = conv_base.layers[-1].output.shape[1:]  # get shape of last layer
    features_shape = (sample_count, *conv_base_output_shape)

    features = np.zeros(shape=features_shape)
    labels = np.zeros(shape=sample_count)
    i = 0
    for inputs_batch, labels_batch in dataset:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        print(i * batch_size, "====>", sample_count)
        if i * batch_size >= sample_count:
            break
    return features, labels

# -------------------------------------------------------------------------------------------------------------
def save_features_labels(dataset_dir: str, features, labels):
    features_file_path = join(dataset_dir, 'features.npy')
    labels_file_path = join(dataset_dir, 'labels.npy')
    np.save(features_file_path, features)
    np.save(labels_file_path, labels)

# -------------------------------------------------------------------------------------------------------------
def load_features_labels(dataset_dir: str):
    features_file_path = join(dataset_dir, 'features.npy')
    labels_file_path = join(dataset_dir, 'labels.npy')
    features = np.load(features_file_path)
    labels = np.load(labels_file_path)
    return features, labels

# -------------------------------------------------------------------------------------------------------------
def for_each_img_dir(dataset_dir: str):
    imgs_dirs_names = [f for f in listdir(dataset_dir) if isdir(join(dataset_dir, f))]
    for img_dir_name in imgs_dirs_names:
        img_dir_path = join(dataset_dir, img_dir_name)
        imgs_names = [f for f in listdir(img_dir_path) if isfile(join(img_dir_path, f))]
        imgs_files_paths = [join(img_dir_path, img_name) for img_name in imgs_names]
        yield img_dir_path, imgs_files_paths

# -------------------------------------------------------------------------------------------------------------
def split_train_validation_datasets(imgs_dir: str, validation_size: float = 0.3):
    train_dir = get_train_dir(imgs_dir)
    validation_dir = get_validation_dir(imgs_dir)
    shutil.rmtree(validation_dir, ignore_errors=True)
    Path(validation_dir).mkdir(parents=True, exist_ok=True)   

    for img_dir_path, imgs_files_path in for_each_img_dir(train_dir):
        n_validations = math.floor(len(imgs_files_path) * validation_size)
        #n_trains = len(imgs_files_path) - n_validations

        img_dir_name = ntpath.basename(img_dir_path)
        new_img_dir_path = join(validation_dir, img_dir_name)
        shutil.rmtree(new_img_dir_path, ignore_errors=True)
        Path(new_img_dir_path).mkdir(parents=True, exist_ok=True)

        print("building directory '{}'...".format(new_img_dir_path))

        validation_imgs_files_path = np.random.choice(imgs_files_path, n_validations, replace=False)
        for old_img_file_path in validation_imgs_files_path:
            img_file_name = ntpath.basename(old_img_file_path)
            new_img_file_path = join(new_img_dir_path, img_file_name)
            print("move image from '{0}' to '{1}'".format(old_img_file_path, new_img_file_path))
            os.rename(old_img_file_path, new_img_file_path)

# -------------------------------------------------------------------------------------------------------------
def use_clf(imgs_dir: str):
    is_extract_features = False
    is_plot_history = True

    train_dir = get_train_dir(imgs_dir)
    validation_dir = get_validation_dir(imgs_dir)

    target_size = (MAX_WORD_W, MAX_WORD_H)
    print('target_size = ', target_size)

    # create a data generator
    shift_side = 0.1
    train_gen = ImageDataGenerator(rotation_range=2, width_shift_range=shift_side, height_shift_range=shift_side)
    train_dataset = train_gen.flow_from_directory(train_dir, target_size=target_size,
                                                  class_mode='binary', batch_size=2)
    validation_gen = ImageDataGenerator()
    validation_dataset = validation_gen.flow_from_directory(validation_dir, target_size=target_size,
                                                  class_mode='binary', batch_size=2)

    #num_of_cls = len(train_dataset.class_indices)
    train_sample_count = len(train_dataset.filenames)
    validation_sample_count = len(validation_dataset.filenames)

    epochs = 10
    batch_size = 2
    learning_rate = 0.0005
    #steps_per_epoch = sample_count // num_of_cls

    input_shape = (*target_size, 3)
    print('input_shape = ', input_shape)


    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    vgg_model.trainable = False
    model = Sequential()
    model.add(vgg_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=learning_rate),
                  metrics=['acc'])

    history = model.fit(train_dataset, steps_per_epoch=100, epochs=epochs,
                                  validation_data=validation_dataset,
                                  validation_steps=50)

    '''
    model = models.Sequential()

    if is_extract_features:
        train_features, train_labels = extract_features(train_dir, train_sample_count, train_dataset,
                                                        batch_size, input_shape, target_size)
        validation_features, validation_labels = extract_features(validation_dir, validation_sample_count, validation_dataset,
                                                        batch_size, input_shape, target_size)
        save_features_labels(train_dir, train_features, train_labels)
        save_features_labels(validation_dir, validation_features, validation_labels)
    else:
        train_features, train_labels = load_features_labels(train_dir)
        validation_features, validation_labels = load_features_labels(validation_dir)

    train_features = np.reshape(train_features, (train_sample_count, 24 * 6 * 512))
    validation_features = np.reshape(validation_features, (validation_sample_count, 24 * 6 * 512))

    model.add(layers.Dense(16, activation='relu', input_dim=24 * 6 * 512))  
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=learning_rate),
                  metrics=['acc'])

    # model.build(input_shape)

    history = model.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_features, validation_labels))
    '''

    # TODO: in fit, use  `validation_steps`


    if is_plot_history:
        epochs = history.epoch
        hyper_params = history.history

        acc = hyper_params['acc']
        val_acc = hyper_params['val_acc']
        loss = hyper_params['loss']
        val_loss = hyper_params['val_loss']

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

# -------------------------------------------------------------------------------------------------------------
def pad_imgs_words(imgs_dir: str, max_word_size = MAX_WORD_SIZE):
    train_dir = get_train_dir(imgs_dir)
    for img_word_file_path, img_word in for_each_img_word(train_dir):
        new_img = ImageOps.pad(img_word.convert("RGB"), size=max_word_size, color=(0xFF, 0xFF, 0xFF))
        img_word.close()
        new_img.save(img_word_file_path)


# endregion

is_full_build_dataset = False
is_fix_img_names = False
is_find_max_train_line_h = False
is_find_max_test_line_h = False
is_build_all_lines_train_dataset = False
is_build_all_lines_test_dataset = False
is_build_all_train_words_dataset = False
is_build_all_test_words_dataset = False
is_find_max_word_size = False
is_pad_imgs_words = False
is_split_train_validation_datasets = False
is_use_clf = True

imgs_dir = LINES_REMOVED_BW_IMG_DIR
train_dir = get_train_dir(imgs_dir)
test_dir = get_test_dir(imgs_dir)

print("Start project")
start_time = time.time()

# region Tasks
if is_full_build_dataset:
    print('building dataset....')
    full_build_dataset(imgs_dir)
    print('dataset is ready to use')

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
    fix_imgs_names()
    print('finish fix images names successfully')

if is_build_all_lines_train_dataset:
    print('building train dataset...')
    build_all_lines_train_dataset(imgs_dir)
    print('train dataset has built successfully')

if is_build_all_lines_test_dataset:
    print('building test dataset...')
    build_all_lines_test_dataset(imgs_dir)
    print('test dataset has built successfully')

if is_build_all_train_words_dataset:
    print('building words of images train dataset...')
    build_all_words_dataset(train_dir)
    print('train images words dataset has built successfully')

if is_build_all_test_words_dataset:
    print('building words of images test dataset...')
    build_all_words_dataset(test_dir)
    print('test images words dataset has built successfully')

if is_find_max_word_size:
    print('find maximum word size...')
    max_w, max_h = find_max_word_size(imgs_dir)
    print('maximum width is', max_w)
    print('maximum height is', max_h)
    print('maximum word size has found successfully')

if is_pad_imgs_words:
    print('padding images words...')
    pad_imgs_words(imgs_dir)
    print('finish pad images words successfully')

if is_split_train_validation_datasets:
    print('splitting images to train and validations directories...')
    split_train_validation_datasets(imgs_dir)
    print('finish split images successfully')

if is_use_clf:
    print('using classifier...')
    use_clf(imgs_dir)
    print('finish use classifier successfully')
# endregion

end_time = time.time()
elapsed_time = end_time - start_time
print('elapsed time = {:.3f} sec'.format(elapsed_time))
print('finish!')
