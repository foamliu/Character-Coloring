# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from skimage import io, color

from data_generator import random_choice, safe_crop
from model import build_encoder_decoder

if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 3

    model_weights_path = 'models/model.12-0.0720.hdf5'
    model = build_encoder_decoder()
    model.load_weights(model_weights_path)

    print(model.summary())

    test_images_folder = 'data/instance-level_human_parsing/Testing/Images'
    id_file = 'data/instance-level_human_parsing/Testing/test_id.txt'
    with open(id_file, 'r') as f:
        names = f.read().splitlines()

    samples = random.sample(names, 10)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_images_folder, image_name + '.jpg')
        bgr = cv.imread(filename)
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        lab = color.rgb2lab(rgb)
        image_size = rgb.shape[:2]

        x, y = random_choice(image_size)
        lab = safe_crop(lab, x, y)
        print('Start processing image: {}'.format(filename))

        x_test = np.empty((1, img_rows, img_cols, 1), dtype=np.float32)
        x_test[0, :, :, 0] = lab[:, :, 0] / 100.

        # L: [0, 100], a: [-86.185, 98.254], b: [-107.863, 94.482].
        ab = model.predict(x_test)
        ab = np.reshape(ab, (img_rows, img_cols, 2))
        a = ab[:, :, 0] * (86.185 + 98.254) - 86.185
        b = ab[:, :, 1] * (107.863 + 94.482) - 107.863

        out = np.empty((img_rows, img_cols, 3), dtype=np.float32)
        out[:, :, 0] = lab[:, :, 0]
        out[:, :, 1:3] = ab
        out = out.astype(np.uint8)
        out = color.lab2rgb(out)

        if not os.path.exists('images'):
            os.makedirs('images')

        gray = (lab[:, :, 0] / 100. * 255.).astype(np.uint8)
        io.imsave('images/{}_image.png'.format(i), gray)
        io.imsave('images/{}_gt.png'.format(i), rgb)
        io.imsave('images/{}_out.png'.format(i), out)

    K.clear_session()
