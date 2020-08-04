import math
import ntpath
import os
import shutil
import time
from os import listdir
from os.path import isfile, join, isdir
from pathlib import Path
from cv2 import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import scipy.io
from PIL import Image, ImageOps
from keras import regularizers
from keras.applications import VGG16, ResNet50, VGG19
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from pytesseract import Output
from tensorflow.keras import layers, optimizers
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from tensorflow.keras.utils import to_categorical

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
IMG_DIRECTORIES = [ORIGINAL_IMG_DIR, LINES_REMOVED_BW_IMG_DIR, LINES_REMOVED_IMG_DIR,
                   IMG_POSITIONS_DIR]  # TODO: do this for each images directories (IMG_POSITIONS_DIR must be the last item)

FEATURES_FILE_NAME = 'features.npy'
LABELS_FILE_NAME = 'labels.npy'

ORIGINAL_IMG_W = 4964
ORIGINAL_IMG_H = 7020
TRAIN_DATASET_RANGE = range(1, 4)  # TODO: change `stop` to 204
MAX_TRAIN_LINE_H = 165
MAX_TEST_LINE_H = 181
MAX_LINE_H = max(MAX_TRAIN_LINE_H, MAX_TEST_LINE_H)
MAX_WORD_W = 776
MAX_WORD_H = 177
REDUCE_WORDS = 4
MAX_WORD_SIZE = (MAX_WORD_W, MAX_WORD_H)
MIN_WORD_W = 40
MIN_WORD_H = 40
DATASET_DIM = (MAX_WORD_W // REDUCE_WORDS, MAX_WORD_H // REDUCE_WORDS)
RELATIVE_SHIFT_IMG_SIZE = 20
IMWRITE_JPEG_QUALITY = 100  # 0 to 100
WHITE_COLOR = (0xFF, 0xFF, 0xFF)


# endregion

# region Functions
# -------------------------------------------------------------------------------------------------------------
def full_build_dataset(imgs_dir: str):
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
    build_all_words_dataset(train_dir, detect_row_num=True)
    print('train images words dataset has built successfully')
    print('building words of images at test dataset...')
    build_all_words_dataset(test_dir, detect_row_num=False)
    print('test images words dataset has built successfully')
    print('find maximum word size...')
    max_word_size = find_max_word_size(imgs_dir)
    print('maximum word size is', max_word_size)
    pad_imgs_words(dataset_dir=train_dir, max_word_size=max_word_size)
    print('train images words dataset has padded successfully')
    print('moving words to train directory...')
    move_words_dir(train_dir)
    print('splitting images to train and validation directories...')
    split_train_validation_datasets(imgs_dir)
    print('padding images words at test directory...')
    pad_imgs_words(dataset_dir=test_dir, max_word_size=max_word_size)
    print('moving words to test directory...')
    move_words_dir(test_dir)
    print('adding data argumentation...')
    add_data_argumentation(train_dir)


# -------------------------------------------------------------------------------------------------------------
def add_data_argumentation(train_dir: str):
    add_shift_imgs(train_dir)
    # TODO: add rotation with 2 degrees & add blur

    # TODO: move data argumentation images to train directory


# -------------------------------------------------------------------------------------------------------------
def add_shift_imgs(train_dir: str):
    shift_left_func = lambda w, h: (- w / RELATIVE_SHIFT_IMG_SIZE, 0)
    shift_right_func = lambda w, h: (w / RELATIVE_SHIFT_IMG_SIZE, 0)
    shift_up_func = lambda w, h: (0, -h / RELATIVE_SHIFT_IMG_SIZE)
    shift_down_func = lambda w, h: (0, h / RELATIVE_SHIFT_IMG_SIZE)
    shift_funcs = {'left': shift_left_func, 'right': shift_right_func, 'up': shift_up_func, 'down': shift_down_func}
    print('adding shift images...')
    imgs_dirs_names = [d for d in listdir(train_dir) if isdir(join(train_dir, d))]
    for img_dir_name in imgs_dirs_names:
        img_dir_path = join(train_dir, img_dir_name)
        imgs_names = [f for f in listdir(img_dir_path) if isfile(join(img_dir_path, f))]
        for img_name in imgs_names:
            img_path = join(img_dir_path, img_name)
            original_img = cv2.imread(img_path)
            original_h, original_w = original_img.shape[:2]
            # cv2.imshow("Originalimage", original_img)
            for shift_func_name, shift_func in shift_funcs.items():
                shift_w, shift_h = shift_func(original_w, original_h)
                M = np.float32([[1, 0, shift_w], [0, 1, shift_h]])
                img_translation = cv2.warpAffine(src=original_img, M=M, dsize=(original_w, original_h),
                                                 borderValue=WHITE_COLOR)
                img_name_without_ex = img_name[:img_name.index('.jpg')]
                img_translation_name = '{}_{}.jpg'.format(img_name_without_ex, shift_func_name)
                img_translation_path = join(img_dir_path, img_translation_name) # TODO: need to be in sub-folder
                cv2.imwrite(img_translation_path, img_translation, [cv2.IMWRITE_JPEG_QUALITY, IMWRITE_JPEG_QUALITY])
                # cv2.imshow('{} translation'.format(shift_func_name), img_translation)
            # cv2.waitKey()
            # cv2.destroyAllWindows()


# -------------------------------------------------------------------------------------------------------------
def move_words_dir(dataset_dir: str):
    imgs_dirs_names = [d for d in listdir(dataset_dir) if isdir(join(dataset_dir, d))]
    for img_dir_name in imgs_dirs_names:
        img_dir_path = join(dataset_dir, img_dir_name)
        imgs_lines_names = [f for f in listdir(img_dir_path) if isfile(join(img_dir_path, f))]
        for img_line_name in imgs_lines_names:
            img_line_path = join(img_dir_path, img_line_name)
            # print("removing image line at '{}'...".format(img_line_path))
            os.remove(img_line_path)
        img_words_path = join(img_dir_path, 'words')
        imgs_words_names = [f for f in listdir(img_words_path) if isfile(join(img_words_path, f))]
        for img_word_name in imgs_words_names:
            old_img_word_path = join(img_words_path, img_word_name)
            new_img_word_path = join(img_dir_path, img_word_name)
            # print("moving image word from '{0}' to '{1}'...".format(old_img_word_path, new_img_word_path))
            os.rename(old_img_word_path, new_img_word_path)
        os.rmdir(img_words_path)


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
def build_all_lines_train_dataset(img_dir: str, max_line_h=MAX_LINE_H):
    train_dir = get_train_dir(img_dir)
    # removing directory
    shutil.rmtree(train_dir, ignore_errors=True)
    # create an empty directory
    Path(train_dir).mkdir(parents=True, exist_ok=True)

    for img_num in TRAIN_DATASET_RANGE:
        build_lines_train_dataset_img(imgs_dir=img_dir, img_num=img_num, max_line_h=max_line_h)


# -------------------------------------------------------------------------------------------------------------
def build_lines_train_dataset_img(imgs_dir: str, img_num: int, max_line_h=MAX_LINE_H):
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
def build_all_lines_test_dataset(imgs_dir: str, max_line_h=MAX_LINE_H):
    test_dir = get_test_dir(imgs_dir)
    # removing directory
    shutil.rmtree(test_dir, ignore_errors=True)
    # create an empty directory
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    for img_num in TRAIN_DATASET_RANGE:
        build_lines_test_dataset_img(img_dir=imgs_dir, img_num=img_num, max_line_h=max_line_h)


# -------------------------------------------------------------------------------------------------------------
def build_lines_test_dataset_img(img_dir: str, img_num: int, max_line_h=MAX_LINE_H):
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
def build_all_words_dataset(dataset_dir: str, detect_row_num: bool):
    for img_num in TRAIN_DATASET_RANGE:
        build_words_dataset_img(dataset_dir=dataset_dir, img_num=img_num, detect_row_num=detect_row_num)


# -------------------------------------------------------------------------------------------------------------
def build_words_dataset_img(dataset_dir: str, img_num: int, detect_row_num: bool):
    print('building image words for image num {}...'.format(img_num))
    img_dir_path = join(dataset_dir, str(img_num))
    lines_files_names = [f for f in listdir(img_dir_path) if isfile(join(img_dir_path, f))]

    img_words_path = join(img_dir_path, 'words')
    shutil.rmtree(img_words_path, ignore_errors=True)
    Path(img_words_path).mkdir(parents=True, exist_ok=True)

    for line_file_name in lines_files_names:
        if detect_row_num:
            start_idx_line_n = line_file_name.index('_') + 1
            end_idx_line_n = line_file_name.index('.')
            row_num = line_file_name[start_idx_line_n:end_idx_line_n]
            img_line_path = join(img_dir_path, '{0}_{1}.jpg'.format(img_num, row_num))
        else:
            row_num = img_num
            img_line_path = join(img_dir_path, '{0}.jpg'.format(img_num))
        img_line = Image.open(img_line_path)
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
def extract_features(sample_count, dataset, batch_size, input_shape, n_of_cls: int):
    conv_base = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    conv_base.summary()

    conv_base_output_shape = conv_base.layers[-1].output.shape[1:]  # get shape of last layer
    features_shape = (sample_count, *conv_base_output_shape)

    features = np.zeros(shape=features_shape)
    labels = to_categorical(np.zeros(shape=sample_count), num_classes=n_of_cls)
    i = 0
    for inputs_batch, labels_batch in dataset:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        print(i * batch_size, "====>", sample_count)
        if i * batch_size >= sample_count:
            break
    return features, labels


# -------------------------------------------------------------------------------------------------------------
def save_features_labels(dataset_dir: str, features, labels):
    features_file_path = join(dataset_dir, FEATURES_FILE_NAME)
    labels_file_path = join(dataset_dir, LABELS_FILE_NAME)
    np.save(features_file_path, features)
    np.save(labels_file_path, labels)


# -------------------------------------------------------------------------------------------------------------
def load_features_labels(dataset_dir: str):
    features_file_path = join(dataset_dir, FEATURES_FILE_NAME)
    labels_file_path = join(dataset_dir, LABELS_FILE_NAME)
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
        # n_trains = len(imgs_files_path) - n_validations

        img_dir_name = ntpath.basename(img_dir_path)
        new_img_dir_path = join(validation_dir, img_dir_name)
        shutil.rmtree(new_img_dir_path, ignore_errors=True)
        Path(new_img_dir_path).mkdir(parents=True, exist_ok=True)

        print("building directory '{}'...".format(new_img_dir_path))

        validation_imgs_files_path = np.random.choice(imgs_files_path, n_validations, replace=False)
        for old_img_file_path in validation_imgs_files_path:
            img_file_name = ntpath.basename(old_img_file_path)
            new_img_file_path = join(new_img_dir_path, img_file_name)
            # print("move image from '{0}' to '{1}'".format(old_img_file_path, new_img_file_path))
            os.rename(old_img_file_path, new_img_file_path)


# -------------------------------------------------------------------------------------------------------------
def shuffle_arrays(arr1, arr2):
    arr_size = arr1.shape[0]
    permutation = np.random.permutation(arr_size) - 1
    return arr1[permutation], arr2[permutation]


# -------------------------------------------------------------------------------------------------------------
def build_model(kernel_regularizer, base_model_dim, learning_rate, n_of_cls):
    model = Sequential()
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=kernel_regularizer, input_dim=base_model_dim))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(n_of_cls, activation='softmax')) # for binary use sigmoid with 1 unit. otherwise use  softmax with number of classes units

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['acc'])
    return model


# -------------------------------------------------------------------------------------------------------------
def train_model(model, train_features, train_labels, epochs, batch_size,
                validation_features, validation_labels, test_features, test_labels):
    history = model.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(validation_features, validation_labels))

    test_loss, test_acc = model.evaluate(test_features, test_labels, batch_size=batch_size)
    print('test_acc =', test_acc)
    print('test_loss =', test_loss)
    return history, test_loss, test_acc


# -------------------------------------------------------------------------------------------------------------
def use_clf():
    is_extract_features = False
    is_plot_history = False
    is_grid_search_regularizer = False

    target_size = DATASET_DIM
    print('target_size = ', target_size)

    batch_size = 5

    # create a data generator
    shift_side = 0.1
    # train_gen = ImageDataGenerator(rotation_range=2, width_shift_range=shift_side, height_shift_range=shift_side)
    train_gen = ImageDataGenerator()
    color_mode = 'rgb' # VGGxx want rgb!
    class_mode = 'categorical' # `binary` for 2 images. otherwise use `categorical`
    train_dataset = train_gen.flow_from_directory(train_dir, target_size=target_size, color_mode=color_mode,
                                                  class_mode=class_mode, batch_size=batch_size)
    validation_gen = ImageDataGenerator()
    validation_dataset = validation_gen.flow_from_directory(validation_dir, target_size=target_size,
                                                            color_mode=color_mode, class_mode=class_mode,
                                                            batch_size=batch_size)
    test_gen = ImageDataGenerator()
    test_dataset = test_gen.flow_from_directory(test_dir, target_size=target_size,
                                                color_mode=color_mode, class_mode=class_mode,
                                                batch_size=batch_size)

    num_of_cls = len(train_dataset.class_indices)
    train_sample_count = len(train_dataset.filenames)
    validation_sample_count = len(validation_dataset.filenames)
    test_sample_count = len(test_dataset.filenames)

    epochs = 10
    learning_rate = 0.0001
    # steps_per_epoch = train_sample_count // num_of_cls

    input_shape = (*target_size, 3)  # 1 for grayscale or 3 for rgb
    print('input_shape = ', input_shape)

    '''
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    '''
    '''
    model = Sequential()
    model.add(layers.Conv2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(layers.MaxPooling2D(8, 8))
    model.add(layers.Conv2D(30, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(4, 4))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(num_of_cls, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=learning_rate),
                  metrics=['acc'])
    history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, batch_size=batch_size,
                        validation_data=validation_dataset)
    '''

    datasets_names = ['train', 'validation', 'test']
    datasets_dirs = [train_dir, validation_dir, test_dir]
    datasets = [train_dataset, validation_dataset, test_dataset]
    sample_counts = [train_sample_count, validation_sample_count, test_sample_count]

    if is_extract_features:
        for dataset_name, dataset_dir, dataset, sample_count in zip(datasets_names, datasets_dirs, datasets,
                                                                    sample_counts):
            print('extract features for {}...'.format(dataset_name))
            features, labels = extract_features(sample_count, dataset, batch_size, input_shape, num_of_cls)
            save_features_labels(dataset_dir, features, labels)
    train_features, train_labels = load_features_labels(train_dir)
    validation_features, validation_labels = load_features_labels(validation_dir)
    test_features, test_labels = load_features_labels(test_dir)

    base_model_dim = 6 * 1 * 512

    train_features = np.reshape(train_features, (train_sample_count, base_model_dim))
    validation_features = np.reshape(validation_features, (validation_sample_count, base_model_dim))
    test_features = np.reshape(test_features, (test_sample_count, base_model_dim))

    train_features, train_labels = shuffle_arrays(train_features, train_labels)
    validation_features, validation_labels = shuffle_arrays(validation_features, validation_labels)

    if is_grid_search_regularizer:
        kernel_regularizers = np.linspace(1e-5, 1e-1, num=7)
        best_acc = 0
        best_regularizer = None
        for kernel_regularizer_val in kernel_regularizers:
            kernel_regularizer = regularizers.l2(kernel_regularizer_val)
            print('checking regularizer {}...'.format(kernel_regularizer_val))
            model = build_model(kernel_regularizer, base_model_dim, learning_rate, num_of_cls)
            history, test_loss, test_acc = train_model(model=model, train_features=train_features,
                                                       train_labels=train_labels,
                                                       epochs=epochs, batch_size=batch_size,
                                                       validation_features=validation_features,
                                                       validation_labels=validation_labels, test_features=test_features,
                                                       test_labels=test_labels)
            if test_acc > best_acc:
                best_acc = test_acc
                best_regularizer = kernel_regularizer_val
        print('best_regularizer =', best_regularizer)
    else:
        kernel_regularizer = regularizers.l2(0.07)
        model = build_model(kernel_regularizer, base_model_dim, learning_rate, num_of_cls)
        model.summary()
        history, test_loss, test_acc = train_model(model=model, train_features=train_features,
                                                   train_labels=train_labels,
                                                   epochs=epochs, batch_size=batch_size,
                                                   validation_features=validation_features,
                                                   validation_labels=validation_labels, test_features=test_features,
                                                   test_labels=test_labels)
        y_pred = model.predict_classes(test_features, batch_size=batch_size)
        y_true = np.argmax(test_labels, axis=1)
        print('Confusion Matrix')
        print(confusion_matrix(y_true, y_pred))
        print('Classification Report')
        target_names = ['Writer {}'.format(i+1) for i in range(num_of_cls)]
        print(classification_report(y_true, y_pred, target_names=target_names))
        avgs = precision_score(y_true, y_pred, average=None)   
        print('avgs =', avgs)
        bad_preds = [True if avg < 0.5 else False for avg in avgs]
        print('bad_preds =', bad_preds)
        n_bad_preds = len(list(filter(lambda is_bad: is_bad, bad_preds)))
        n_good_preds = len(avgs) - n_bad_preds
        weighted_avg = n_good_preds / len(avgs)
        print('weighted_avg =', weighted_avg)


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
def pad_imgs_words(dataset_dir: str, max_word_size=MAX_WORD_SIZE):
    for img_word_file_path, img_word in for_each_img_word(dataset_dir):
        new_img = ImageOps.pad(img_word.convert("RGB"), size=max_word_size, color=(0xFF, 0xFF, 0xFF))
        img_word.close()
        new_img.save(img_word_file_path)


# endregion


is_full_build_dataset = False
is_use_clf = True

imgs_dir = LINES_REMOVED_IMG_DIR
train_dir = get_train_dir(imgs_dir)
validation_dir = get_validation_dir(imgs_dir)
test_dir = get_test_dir(imgs_dir)

print("Start project")
start_time = time.time()

# region Tasks
if is_full_build_dataset:
    print('building dataset....')
    full_build_dataset(imgs_dir)
    print('dataset is ready to use')

if is_use_clf:
    print('using classifier...')
    use_clf()
    print('finish use classifier successfully')
# endregion

end_time = time.time()
elapsed_time = end_time - start_time
print('elapsed time = {:.3f} sec'.format(elapsed_time))
print('finish!')
