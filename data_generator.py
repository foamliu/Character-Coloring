import os
import random

import cv2 as cv
import numpy as np
import sklearn.neighbors as nn
from keras.utils import Sequence

from config import batch_size, img_rows, img_cols, nb_neighbors

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


def get_soft_encoding(X, nn_finder, nb_q):
    sigma_neighbor = 5

    # Get the distance to and the idx of the nearest neighbors
    dist_neighb, idx_neigh = nn_finder.kneighbors(X)

    # Smooth the weights with a gaussian kernel
    wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]

    # format the target
    Y = np.zeros((X.shape[0], nb_q))
    idx_pts = np.arange(X.shape[0])[:, np.newaxis]
    Y[idx_pts, idx_neigh] = wts

    return Y


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

        # Load the array of quantized ab value
        q_ab = np.load("data/pts_in_hull.npy")
        self.nb_q = q_ab.shape[0]
        # Fit a NN to q_ab
        self.nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 1), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i]
            filename = os.path.join(self.images_folder, name + '.jpg')
            image = cv.imread(filename)
            image_size = image.shape[:2]
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            x, y = random_choice(image_size)
            image = safe_crop(image, x, y)
            gray = safe_crop(gray, x, y)

            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                gray = np.fliplr(gray)

            x = gray / 255.
            y = image / 255.

            batch_x[i_batch, :, :, 0] = x
            batch_y[i_batch, :, :, 0:3] = y

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
