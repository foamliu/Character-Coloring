# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from data_generator import random_choice, safe_crop
from model import build_encoder_decoder

if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 3

    model_weights_path = 'models/model.13-0.0052.hdf5'
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
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        image_size = bgr.shape[:2]

        x, y = random_choice(image_size)
        bgr = safe_crop(bgr, x, y)
        lab = safe_crop(lab, x, y)
        print('Start processing image: {}'.format(filename))

        x_test = np.empty((1, img_rows, img_cols, 1), dtype=np.float32)
        x_test[0, :, :, 0] = lab[:, :, 0] / 255.

        # L: [0, 100], a: [-86.185, 98.254], b: [-107.863, 94.482].
        ab = model.predict(x_test)
        ab = np.reshape(ab, (img_rows, img_cols, 2))
        a = ab[:, :, 0] * 184 + 42
        b = ab[:, :, 1] * 203 + 20

        out = np.empty((img_rows, img_cols, 3), dtype=np.float32)
        out[:, :, 0] = lab[:, :, 0]
        out[:, :, 1] = a
        out[:, :, 2] = b
        out = out.astype(np.uint8)
        out = cv.cvtColor(bgr, cv.COLOR_LAB2BGR)

        if not os.path.exists('images'):
            os.makedirs('images')

        bgr = bgr.astype(np.uint8)
        gray = (lab[:, :, 0]).astype(np.uint8)
        cv.imwrite('images/{}_image.png'.format(i), gray)
        cv.imwrite('images/{}_gt.png'.format(i), bgr)
        cv.imwrite('images/{}_out.png'.format(i), out)

    K.clear_session()
