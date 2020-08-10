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
from keras.applications import VGG16, ResNet50, VGG19, DenseNet201
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from pytesseract import Output
from tensorflow.keras import layers, optimizers
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from tensorflow.keras.utils import to_categorical
import pickle
import random

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
IMG_DIRECTORIES = [ORIGINAL_IMG_DIR, LINES_REMOVED_BW_IMG_DIR, LINES_REMOVED_IMG_DIR,
                   IMG_POSITIONS_DIR]  # TODO: do this for each images directories (IMG_POSITIONS_DIR must be the last item)

FEATURES_FILE_NAME = 'features.npy'
LABELS_FILE_NAME = 'labels.npy'
FILES_PATHS_FILE_NAME = 'files_paths.pickle'

MODEL_FILE_NAME = 'model'

ORIGINAL_IMG_W = 4964
ORIGINAL_IMG_H = 7020
TRAIN_DATASET_RANGE = range(1, 21)  # TODO: change `stop` to 204
MAX_TRAIN_LINE_H = 170
MAX_TEST_LINE_H = 181
MAX_LINE_H = max(MAX_TRAIN_LINE_H, MAX_TEST_LINE_H)
MAX_WORD_W = 1306
MAX_WORD_H = 332
REDUCE_WORDS = 4
MIN_ROTATE_ANGLE = 2
MAX_ROTATE_ANGLE = 7
ROTATE_RANGE = range(MIN_ROTATE_ANGLE, MAX_ROTATE_ANGLE)
MAX_WORD_SIZE = (MAX_WORD_W, MAX_WORD_H)
MIN_WORD_W = 80
MIN_WORD_H = 50
DATASET_DIM = (MAX_WORD_W // REDUCE_WORDS, MAX_WORD_H // REDUCE_WORDS)
REL_SHIFT_IMG_SIZE_RANGE = range(20, 40)
BLUR_SIZE_RANGE = range(2, 10)
SCALE_PERCENT_RANGE = range(1, 10)
IMWRITE_JPEG_QUALITY = 100  # 0 to 100
WHITE_COLOR = (0xFF, 0xFF, 0xFF)
N_IMAGES_TO_SHOW = 9
MIN_WORDS_TEST_DATASET = 5


# endregion

# region Functions
# -------------------------------------------------------------------------------------------------------------
def full_build_dataset(imgs_dir: str):
    # print('fixing images names...')
    # fix_imgs_names()
    print('building words of images at train dataset...')
    build_direct_imgs_words(imgs_dir)
    print('train and test images words dataset has built successfully')
    max_word_size = find_max_word_size()
    print('maximum word size is', max_word_size)
    print('padding images words at train directory...')
    pad_imgs_words(dataset_dir=train_dir, max_word_size=max_word_size)
    print('train images words dataset has padded successfully')
    print('padding images words at test directory...')
    pad_imgs_words(dataset_dir=test_dir, max_word_size=max_word_size)
    print('test images words dataset has padded successfully')
    print('splitting images to train and validation directories...')
    #split_train_validation_datasets(imgs_dir)
    print('adding data argumentation...')
    # add_data_argumentation(train_dir)

# -------------------------------------------------------------------------------------------------------------
def for_writer_img(imgs_dir: str):
    for img_num in TRAIN_DATASET_RANGE:
        img_name = '{}.jpg'.format(img_num)
        img_file_path = join(imgs_dir, img_name)
        img = cv2.imread(img_file_path)
        yield img_num, img

# -------------------------------------------------------------------------------------------------------------
def build_direct_imgs_words(imgs_dir: str):
    shutil.rmtree(train_dir, ignore_errors=True)
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(test_dir, ignore_errors=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    for img_num, img in for_writer_img(imgs_dir):
        img_train_dir_path = join(train_dir, str(img_num))
        img_test_dir_path = join(test_dir, str(img_num))
        print('working on image number {}...'.format(img_num))
        line_info = load_img_lines_info(img_num=img_num)
        img_check_file_path = join(LINES_REMOVED_BW_IMG_DIR, '{}.jpg'.format(img_num))
        img_check = cv2.imread(img_check_file_path)
        build_direct_train_img_words(img_num, img_train_dir_path, img.copy(), img_check.copy(), line_info)
        build_direct_test_img_words(img_num, img_test_dir_path, img.copy(), img_check.copy(), line_info)

# -------------------------------------------------------------------------------------------------------------
def build_direct_train_img_words(img_num, img_dir_path, img, img_check, line_info):  
    train_img = crop_train_img(line_info, img)
    train_img_check = crop_train_img(line_info, img)
    find_save_words(img_num, img_dir_path, train_img, train_img_check)

# -------------------------------------------------------------------------------------------------------------
def crop_train_img(line_info, img):
    y_positions = parse_y_positions(line_info)
    top_test_area = line_info['top_test_area'].flatten()[0]
    bottom_test_area = line_info['bottom_test_area'].flatten()[0]
    img_no_test = cv2.rectangle(img, (0, top_test_area), (ORIGINAL_IMG_W, bottom_test_area), WHITE_COLOR, cv2.FILLED)
    start_h = y_positions[0]
    end_h = y_positions[-1]
    crop_img = img_no_test[start_h: end_h, 0: ORIGINAL_IMG_W]
    return crop_img

# -------------------------------------------------------------------------------------------------------------
def build_direct_test_img_words(img_num, img_dir_path, img, img_check, line_info):
    top_test_area = line_info['top_test_area'].flatten()[0]
    bottom_test_area = line_info['bottom_test_area'].flatten()[0]
    crop_img = img[top_test_area: bottom_test_area, 0: ORIGINAL_IMG_W]
    crop_img_check = img_check[top_test_area: bottom_test_area, 0: ORIGINAL_IMG_W]
    find_save_words(img_num, img_dir_path, crop_img, crop_img_check)

# -------------------------------------------------------------------------------------------------------------
def find_save_words(img_num: int, img_dir_path: str, img, img_check):
    shutil.rmtree(img_dir_path, ignore_errors=True)
    Path(img_dir_path).mkdir(parents=True, exist_ok=True)
    img_data = pytesseract.image_to_data(cv2.cvtColor(img_check, cv2.COLOR_BGR2RGB), lang='heb', output_type=Output.DICT)
    n_boxes = len(img_data['word_num'])
    n_words = 0
    for i in range(n_boxes):
        if img_data['word_num'][i] > 0:
            (x, y, w, h) = (img_data['left'][i], img_data['top'][i], img_data['width'][i],
                            img_data['height'][i])
            
            word_img = img[y : y + h, x : x + w]

            if is_white_img(Image.fromarray(cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB))) or (w < MIN_WORD_W or h < MIN_WORD_H):
                continue

            word_img_name = '{0}_{1}.jpg'.format(img_num, i)
            word_img_path = join(img_dir_path, word_img_name)
            cv2.imwrite(word_img_path, word_img, [cv2.IMWRITE_JPEG_QUALITY, IMWRITE_JPEG_QUALITY])
            n_words += 1
            #img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0xFF, 0), 2)

    '''   
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    '''
    


# -------------------------------------------------------------------------------------------------------------
def add_data_argumentation(train_dir: str):
    shift_left_func = lambda w, h: (- w / random.choice(REL_SHIFT_IMG_SIZE_RANGE), 0)
    shift_right_func = lambda w, h: (w / random.choice(REL_SHIFT_IMG_SIZE_RANGE), 0)
    shift_up_func = lambda w, h: (0, -h / random.choice(REL_SHIFT_IMG_SIZE_RANGE))
    shift_down_func = lambda w, h: (0, h / random.choice(REL_SHIFT_IMG_SIZE_RANGE))
    rotate_left_func = lambda w, h, angle, var: (w // 2, h // 2, -angle , var)
    rotate_right_func = lambda w, h, angle, var: (w // 2, h // 2, angle , var)
    rotate_funcs = {'rotate_left': rotate_left_func, 'rotate_right': rotate_right_func}
    shift_funcs = {'left': shift_left_func, 'right': shift_right_func, 'up': shift_up_func, 'down': shift_down_func}
    imgs_dirs_names = [d for d in listdir(train_dir) if isdir(join(train_dir, d))]
    for img_dir_name in imgs_dirs_names:
        print('scanning image number {}...'.format(img_dir_name))
        img_dir_path = join(train_dir, img_dir_name)
        imgs_names = [f for f in listdir(img_dir_path) if isfile(join(img_dir_path, f))]
        for img_name in imgs_names:
            img_path = join(img_dir_path, img_name)
            original_img = cv2.imread(img_path)
            original_h, original_w = original_img.shape[:2]
            # cv2.imshow("Originalimage", original_img)
            for shift_func_name, shift_func in shift_funcs.items():
                if bool(random.getrandbits(1)):
                    shift_w, shift_h = shift_func(original_w, original_h)
                    M = np.float32([[1, 0, shift_w], [0, 1, shift_h]])
                    img_translation = cv2.warpAffine(src=original_img, M=M, dsize=(original_w, original_h),
                                                    borderValue=WHITE_COLOR)
                    img_name_without_ex = img_name[:img_name.index('.jpg')]
                    img_translation_name = '{}_{}.jpg'.format(img_name_without_ex, shift_func_name)
                    img_translation_path = join(img_dir_path, img_translation_name)
                    cv2.imwrite(img_translation_path, img_translation, [cv2.IMWRITE_JPEG_QUALITY, IMWRITE_JPEG_QUALITY])
            angle = random.choice(ROTATE_RANGE)
            var = 1.0
            for rotate_func_name, rotate_func in rotate_funcs.items():
                if bool(random.getrandbits(1)):
                    rot_w, rot_h, rot_angle, v = rotate_func(original_w, original_h, angle, var)
                    M = cv2.getRotationMatrix2D((rot_w, rot_h), rot_angle, v)
                    img_translation = cv2.warpAffine(src=original_img, M=M, dsize=(original_w, original_h),
                                                    borderValue=WHITE_COLOR)
                    img_name_without_ex = img_name[:img_name.index('.jpg')]
                    img_translation_name = '{}_{}.jpg'.format(img_name_without_ex, rotate_func_name)
                    img_translation_path = join(img_dir_path, img_translation_name)
                    cv2.imwrite(img_translation_path, img_translation, [cv2.IMWRITE_JPEG_QUALITY, IMWRITE_JPEG_QUALITY])

            blur_size = random.choice(BLUR_SIZE_RANGE)
            blur = cv2.blur(original_img, (blur_size, blur_size))
            img_name_without_ex = img_name[:img_name.index('.jpg')]
            img_translation_name = '{}_{}.jpg'.format(img_name_without_ex, 'blur')
            img_translation_path = join(img_dir_path, img_translation_name)
            cv2.imwrite(img_translation_path, blur, [cv2.IMWRITE_JPEG_QUALITY, IMWRITE_JPEG_QUALITY])

            scale_percent = random.choice(SCALE_PERCENT_RANGE)
            scale_factor = scale_percent / 100
            scale_start_w = int(original_w * scale_factor)
            scale_end_w = original_w - scale_start_w
            scale_start_h = int(original_h * scale_factor)
            scale_end_h = original_h - scale_start_h
            cropped_scaled_img = original_img[scale_start_h : scale_end_h, scale_start_w : scale_end_w]
            resized_img = cv2.resize(src=cropped_scaled_img, dsize=(original_w, original_h), interpolation=cv2.INTER_AREA)
            resized_img_name = '{}_{}.jpg'.format(img_name_without_ex, 'scale')
            resized_img_path = join(img_dir_path, resized_img_name)
            cv2.imwrite(resized_img_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, IMWRITE_JPEG_QUALITY])
            


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
        # print('checking image number ', img_num)
        line_info = load_img_lines_info(img_num=img_num)
        y_positions = parse_y_positions(line_info)
        for idx in range(len(y_positions) - 1):
            line_h = y_positions[idx + 1] - y_positions[idx]
            lines_h.append(line_h)
    # print('lines height = ', lines_h)
    return max(lines_h)


# -------------------------------------------------------------------------------------------------------------
def find_max_test_line_h():
    lines_h = []
    for img_num in TRAIN_DATASET_RANGE:
        # print('checking image number ', img_num)
        line_info = load_img_lines_info(img_num)
        top_test_area = line_info['top_test_area'].flatten()[0]
        bottom_test_area = line_info['bottom_test_area'].flatten()[0]
        line_h = bottom_test_area - top_test_area
        lines_h.append(line_h)
    # print('lines height = ', lines_h)
    return max(lines_h)


# -------------------------------------------------------------------------------------------------------------
def show_9_images(title, the_images, the_labels, images_paths):
    if the_images.size == 0:
        return

    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle(title, fontsize=16)

    L = len(the_labels)

    k = 0
    for i in range(0, 3):
        for j in range(0, 3):
            img = the_images[k, :]
            label = the_labels[k]
            img_path = images_paths[k]
            img_name = os.path.basename(img_path)

            axs[i, j].imshow(img)
            axs[i, j].title.set_text('{} | {}'.format(label, img_name))
            axs[i, j].axis('off')
            k += 1
            if k >= L:
                plt.show()
                return

    plt.show()

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
def for_each_img_word(dataset_dir: str):
    imgs_dirs_names = [d for d in listdir(dataset_dir) if isdir(join(dataset_dir, d))]
    for img_dir_name in imgs_dirs_names:
        print('scanning image number {}...'.format(img_dir_name))
        img_dir = join(dataset_dir, img_dir_name)
        imgs_words_names = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
        for img_word_name in imgs_words_names:
            img_word_file_path = join(img_dir, img_word_name)
            img_word = Image.open(img_word_file_path)
            yield img_word_file_path, img_word


# -------------------------------------------------------------------------------------------------------------
def find_max_word_size():
    max_w, max_h = (0, 0)
    dataset_dirs = [train_dir, test_dir]
    for dataset_dir in dataset_dirs:
        for _, img_word in for_each_img_word(dataset_dir):
            img_w, img_h = img_word.size
            max_w = max(img_w, max_w)
            max_h = max(img_h, max_h)
    return max_w, max_h


# -------------------------------------------------------------------------------------------------------------
def extract_features(sample_count, dataset, batch_size, input_shape, n_of_cls: int):
    conv_base = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)

    conv_base.summary()

    files_paths = [join(dataset.directory, file_name) for file_name in dataset.filenames]

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
    
    return features, labels, files_paths


# -------------------------------------------------------------------------------------------------------------
def save_features_labels(dataset_dir: str, features, labels, files_paths):
    features_file_path = join(dataset_dir, FEATURES_FILE_NAME)
    labels_file_path = join(dataset_dir, LABELS_FILE_NAME)
    files_paths_path = join(dataset_dir, FILES_PATHS_FILE_NAME)
    np.save(features_file_path, features)
    np.save(labels_file_path, labels)
    with open(files_paths_path,  'wb') as f:
        pickle.dump(files_paths, f)


# -------------------------------------------------------------------------------------------------------------
def load_features_labels(dataset_dir: str):
    features_file_path = join(dataset_dir, FEATURES_FILE_NAME)
    labels_file_path = join(dataset_dir, LABELS_FILE_NAME)
    files_paths_path = join(dataset_dir, FILES_PATHS_FILE_NAME)
    features = np.load(features_file_path)
    labels = np.load(labels_file_path)
    with open(files_paths_path,  'rb') as f:
        files_paths = pickle.load(f)
    return features, labels, files_paths


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
def shuffle_arrays(arr1, arr2, python_list = None):
    arr_size = arr1.shape[0]
    permutation = np.random.permutation(arr_size) - 1
    new_arr1 = arr1[permutation]
    new_arr2 = arr2[permutation]
    if python_list is not None:
        new_python_list = np.array(python_list, dtype='S')[permutation].tolist()
        return new_arr1, new_arr2, new_python_list
    return new_arr1, new_arr2


# -------------------------------------------------------------------------------------------------------------
def build_model(kernel_regularizer, base_model_dim, learning_rate, n_of_cls: int):
    bias_regularizer = None # regularizers.l2(1e-8)
    activity_regularizer = None # regularizers.l2(1e-8)
    model = Sequential()
    #TODO: try max pooling for better performance
    #model.add(layers.MaxPool2D(pool_size=(4, 4)))
    # dropout_rate = 0.3
    units = 256
    # dropout_rate = 0.05
    model.add(layers.Dense(units, activation='relu', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                             activity_regularizer=activity_regularizer,input_dim=base_model_dim))
    #model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(units, activation='relu', kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer, bias_regularizer=bias_regularizer))
    #model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(units, activation='relu', kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer, bias_regularizer=bias_regularizer))
    #model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(units, activation='relu', kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer, bias_regularizer=bias_regularizer))
    model.add(layers.Dropout(0.2))
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
def get_writers_numbers(labels):
    writers_nums = np.argmax(labels, axis=1)
    writers_nums[writers_nums == 1] = 10
    writers_nums[writers_nums == 0] = 1
    return writers_nums


# -------------------------------------------------------------------------------------------------------------
def load_model():
    model_file_path = join(imgs_dir, MODEL_FILE_NAME)
    return keras.models.load_model(model_file_path)

# -------------------------------------------------------------------------------------------------------------
def save_model(model):
    model_file_path = join(imgs_dir, MODEL_FILE_NAME)
    model.save(model_file_path)

# -------------------------------------------------------------------------------------------------------------
def use_clf():
    is_extract_features = False
    is_plot_history = False
    is_grid_search_regularizer = False
    is_show_wrong_pred_imgs = True
    is_show_dataset_imgs = False
    is_train_model = False

    target_size = DATASET_DIM
    print('target_size = ', target_size)

    batch_size = 20

    # create a data generator
    # shift_side = 0.1
    # train_gen = ImageDataGenerator(rotation_range=2, width_shift_range=shift_side, height_shift_range=shift_side)
    train_gen = ImageDataGenerator()
    color_mode = 'rgb' # VGGxx want rgb!
    class_mode = 'categorical' # `binary` for 2 images. otherwise use `categorical`
    train_dataset = train_gen.flow_from_directory(train_dir, target_size=target_size, color_mode=color_mode,
                                                  class_mode=class_mode, batch_size=batch_size, shuffle=False)
    validation_gen = ImageDataGenerator()
    validation_dataset = validation_gen.flow_from_directory(validation_dir, target_size=target_size,
                                                            color_mode=color_mode, class_mode=class_mode,
                                                            batch_size=batch_size, shuffle=False)
    test_gen = ImageDataGenerator()
    test_dataset = test_gen.flow_from_directory(test_dir, target_size=target_size,
                                                color_mode=color_mode, class_mode=class_mode,
                                                batch_size=batch_size, shuffle=False)

    

    num_of_cls = len(train_dataset.class_indices)
    train_sample_count = len(train_dataset.filenames)
    validation_sample_count = len(validation_dataset.filenames)
    test_sample_count = len(test_dataset.filenames)

    epochs = 8
    learning_rate = 0.0001
    # steps_per_epoch = train_sample_count // num_of_cls

    input_shape = (*target_size, 3)  # 1 for grayscale or 3 for rgb
    print('input_shape = ', input_shape)

    datasets_names = ['train', 'validation', 'test']
    datasets_dirs = [train_dir, validation_dir, test_dir]
    datasets = [train_dataset, validation_dataset, test_dataset]
    sample_counts = [train_sample_count, validation_sample_count, test_sample_count]

    if is_extract_features:
        for dataset_name, dataset_dir, dataset, sample_count in zip(datasets_names, datasets_dirs, datasets,
                                                                    sample_counts):
            print('extract features for {}...'.format(dataset_name))
            features, labels, files_paths = extract_features(sample_count, dataset, batch_size, input_shape, num_of_cls)
            save_features_labels(dataset_dir, features, labels, files_paths)
    train_features, train_labels, train_files_paths = load_features_labels(train_dir)
    validation_features, validation_labels, validation_files_paths = load_features_labels(validation_dir)
    test_features, test_labels, test_files_paths = load_features_labels(test_dir)

    base_model_dim = 10 * 2 * 1920

    train_features = np.reshape(train_features, (train_sample_count, base_model_dim))
    validation_features = np.reshape(validation_features, (validation_sample_count, base_model_dim))
    test_features = np.reshape(test_features, (test_sample_count, base_model_dim))

    train_features, train_labels, train_files_paths = shuffle_arrays(train_features, train_labels, train_files_paths)
    validation_features, validation_labels, validation_files_paths = shuffle_arrays(validation_features, validation_labels, validation_files_paths)

    datasets_files_paths = [train_files_paths, validation_files_paths, test_files_paths]

    train_labels_numbers = get_writers_numbers(train_labels)
    validation_numbers = get_writers_numbers(validation_labels)
    test_labels_numbers = get_writers_numbers(test_labels)

    labels_numbers = [train_labels_numbers, validation_numbers, test_labels_numbers]

    if is_show_dataset_imgs:
        for dataset_name, dataset_labels_numbers, dataset_files_paths in zip(datasets_names, labels_numbers, datasets_files_paths):
            images_paths = [fp for i, fp in enumerate(dataset_files_paths) if i < N_IMAGES_TO_SHOW]
            images = np.asarray([np.asarray(Image.open(fp)) for fp in images_paths])
            writers_labels_nums = ['Writer {}'.format(n) for n in dataset_labels_numbers[:N_IMAGES_TO_SHOW]]
            show_9_images('{} images'.format(dataset_name), images, writers_labels_nums, images_paths)

    if is_grid_search_regularizer:
        kernel_regularizers = np.linspace(1e-10, 1e-1, num=3)
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
        kernel_regularizer_val = 1e-7
        kernel_regularizer = regularizers.l2(kernel_regularizer_val)
        if not is_train_model:
            model = load_model()
        else:
            model = build_model(kernel_regularizer, base_model_dim, learning_rate, num_of_cls)
            model.summary()
            history, test_loss, test_acc = train_model(model=model, train_features=train_features,
                                                    train_labels=train_labels,
                                                    epochs=epochs, batch_size=batch_size,
                                                    validation_features=validation_features,
                                                    validation_labels=validation_labels, test_features=test_features,
                                                    test_labels=test_labels)
            save_model(model)
        y_pred = model.predict_classes(test_features, batch_size=batch_size)
        y_true = np.argmax(test_labels, axis=1)
        words_c_matrix = np.zeros(shape=(num_of_cls, num_of_cls), dtype=int)
        for pred_val, true_val in zip(y_pred, y_true):
            words_c_matrix[true_val, pred_val] += 1

        #print('y_pred =', y_pred)
        #print('y_true =', y_true)
        print('words_c_matrix:\n', words_c_matrix)
        lines_c_matrix = np.zeros(shape=(num_of_cls, num_of_cls), dtype=int)
        for i in range(num_of_cls):
            writer_label = np.argmax(words_c_matrix[i]).item(0)
            lines_c_matrix[i, writer_label] = 1
        print('lines_c_matrix:\n', lines_c_matrix)
        
        n_good_lines_preds = np.sum(lines_c_matrix.diagonal())
        test_lines_acc = n_good_lines_preds / num_of_cls
        print('n_good_lines_preds = ', n_good_lines_preds)
        print('test_lines_acc = ', test_lines_acc)

        if is_show_wrong_pred_imgs:
            bad_preds_subarr = []
            true_subarr = []
            images_paths_subarr = []
            rand_y_pred, rand_y_true, rand_files_path = shuffle_arrays(y_pred, y_true, test_files_paths)
            for idx, (pred_val, true_val) in enumerate(zip(rand_y_pred, rand_y_true)):
                if true_val != pred_val:
                    bad_preds_subarr.append(pred_val)
                    true_subarr.append(true_val)
                    img_path = rand_files_path[idx]
                    images_paths_subarr.append(img_path)
            
            n_wrong_preds_show = min(len(images_paths_subarr), N_IMAGES_TO_SHOW)
            images_paths = [fp for i, fp in enumerate(images_paths_subarr) if i < n_wrong_preds_show]
            images = np.asarray([np.asarray(Image.open(fp)) for fp in images_paths])
            bad_preds_subarr = bad_preds_subarr[:n_wrong_preds_show]
            true_subarr = true_subarr[:n_wrong_preds_show]
            # Note: 0 => writer 1, 1 => writer 10, 2 => writer 2, ...
            labels = ['True = {} | Pred = {}'.format(t, p) for p, t in zip(bad_preds_subarr, true_subarr)]
            show_9_images('Wrong words predictions', images, labels, images_paths)


            '''
            for i in range(num_of_cls):
                if lines_c_matrix[i, i] == 0:
                    pred_writer_label = np.argmax(lines_c_matrix[i, :])
                    true_writer_label = i
                    print('pred_writer_label =', pred_writer_label)
                    print('true_writer_label =', true_writer_label)
            '''

        if is_train_model and is_plot_history:
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


is_full_build_dataset = True
is_use_clf = False

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
