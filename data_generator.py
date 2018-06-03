import os
import random

import cv2 as cv
import numpy as np
from keras.utils import Sequence

from config import batch_size, img_rows, img_cols

train_images_folder = 'data/instance-level_human_parsing/Training/Images'
train_categories_folder = 'data/instance-level_human_parsing/Training/Categories'
valid_images_folder = 'data/instance-level_human_parsing/Validation/Images'
valid_categories_folder = 'data/instance-level_human_parsing/Validation/Categories'


def random_choice(image_size):
    height, width = image_size
    crop_height, crop_width = 320, 320
    x = random.randint(0, max(0, width - crop_width))
    y = random.randint(0, max(0, height - crop_height))
    return x, y


def safe_crop(mat, x, y):
    crop_height, crop_width = 320, 320
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    return ret


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        if usage == 'train':
            id_file = 'data/instance-level_human_parsing/Training/train_id.txt'
            self.images_folder = train_images_folder
            self.categories_folder = train_categories_folder
        else:
            id_file = 'data/instance-level_human_parsing/Validation/val_id.txt'
            self.images_folder = valid_images_folder
            self.categories_folder = valid_categories_folder

        with open(id_file, 'r') as f:
            self.names = f.read().splitlines()

        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 1), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols, 2), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i]
            filename = os.path.join(self.images_folder, name + '.jpg')
            # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
            bgr = cv.imread(filename)
            # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
            lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
            image_size = lab.shape[:2]

            x, y = random_choice(image_size)
            lab = safe_crop(lab, x, y)

            if np.random.random_sample() > 0.5:
                lab = np.fliplr(lab)

            x = lab[:, :, 0] / 255.
            y_a = (lab[:, :, 1] - 42) / 184
            y_b = (lab[:, :, 2] - 20) / 203

            batch_x[i_batch, :, :, 0] = x
            batch_y[i_batch, :, :, 0] = y_a
            batch_y[i_batch, :, :, 1] = y_b

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
