import multiprocessing

import cv2 as cv
import keras.backend as K
from tensorflow.python.client import device_lib

from config import epsilon_sqr


def prediction_loss(y_true, y_pred):
    c_g = y_true[:, :, :, 0:2]
    c_p = y_pred[:, :, :, 0:2]
    diff = c_p - c_g
    return K.mean(K.square(diff))


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
