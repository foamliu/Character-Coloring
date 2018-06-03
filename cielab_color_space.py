import cv2 as cv
import numpy as np

if __name__ == '__main__':
    L = [0] * 256 ** 3
    a = [0] * 256 ** 3
    b = [0] * 256 ** 3
    i = 0
    for r in range(256):
        for g in range(256):
            for bb in range(256):
                im = np.array((bb, g, r), np.uint8).reshape(1, 1, 3)
                cv.cvtColor(im, cv.COLOR_BGR2LAB, im)  # tranform it to LAB
                L[i] = im[0, 0, 0]
                a[i] = im[0, 0, 1]
                b[i] = im[0, 0, 2]
                i += 1

    print(min(L), '<=L<=', max(L))
    print(min(a), '<=a<=', max(a))
    print(min(b), '<=b<=', max(b))
