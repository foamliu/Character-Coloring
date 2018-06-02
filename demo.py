# import the necessary packages
import os
import random

import cv2 as cv
from skimage import io, color
import keras.backend as K
import numpy as np

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
        rgb = io.imread(filename)
        lab = color.rgb2lab(rgb)
        image_size = rgb.shape[:2]

        x, y = random_choice(image_size)
        lab = safe_crop(lab, x, y)
        gray = lab[:, :, 0]
        print('Start processing image: {}'.format(filename))

        x_test = np.empty((1, img_rows, img_cols, 1), dtype=np.float32)
        x_test[0, :, :, 0] = gray / 100.

        ab = model.predict(x_test)
        ab = np.reshape(ab, (img_rows, img_cols, 2))
        ab = (ab - 0.5) * 127.

        out = np.empty((img_rows, img_cols, 3), dtype=np.float32)
        out[:, :, 0] = gray
        out[:, :, 1:3] = ab
        out = out.astype(np.uint8)
        out = color.lab2rgb(out)

        if not os.path.exists('images'):
            os.makedirs('images')

        gray = (gray / 100. * 255.).astype(np.uint8)
        io.imsave('images/{}_image.png'.format(i), gray)
        io.imsave('images/{}_gt.png'.format(i), rgb)
        io.imsave('images/{}_out.png'.format(i), out)

    K.clear_session()
