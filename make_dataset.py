import os

import cv2 as cv
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
import sklearn.neighbors as nn
from matplotlib.colors import LogNorm
from skimage import color


def load_data(size=64):
    images_folder = 'data/instance-level_human_parsing/Training/Images'
    names = [f for f in os.listdir(images_folder) if f.lower().endswith('.jpg')]
    num_samples = len(names)
    X_ab = np.empty((num_samples, size, size, 2))
    for i in range(num_samples):
        name = names[i]
        filename = os.path.join(images_folder, name)
        bgr = cv.imread(filename)
        bgr = cv.resize(bgr, (size, size), cv.INTER_CUBIC)
        rgb = bgr[:, :, ::-1]
        lab = color.rgb2lab(rgb)
        X_ab[i] = lab[:, :, 1:]
    return X_ab


def compute_color_prior(X_ab, size=64, do_plot=False):
    # Load the gamut points location
    q_ab = np.load(os.path.join(data_dir, "pts_in_hull.npy"))

    if do_plot:
        plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])
        for i in range(q_ab.shape[0]):
            ax.scatter(q_ab[:, 0], q_ab[:, 1])
            ax.annotate(str(i), (q_ab[i, 0], q_ab[i, 1]), fontsize=6)
            ax.set_xlim([-110, 110])
            ax.set_ylim([-110, 110])

    npts, c, h, w = X_ab.shape
    X_a = np.ravel(X_ab[:, 0, :, :])
    X_b = np.ravel(X_ab[:, 1, :, :])
    X_ab = np.vstack((X_a, X_b)).T

    if do_plot:
        plt.hist2d(X_ab[:, 0], X_ab[:, 1], bins=100, norm=LogNorm())
        plt.xlim([-110, 110])
        plt.ylim([-110, 110])
        plt.colorbar()
        plt.show()
        plt.clf()
        plt.close()

    # Create nearest neighbord instance with index = q_ab
    NN = 1
    nearest = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(q_ab)
    # Find index of nearest neighbor for X_ab
    dists, ind = nearest.kneighbors(X_ab)

    # We now count the number of occurrences of each color
    ind = np.ravel(ind)
    counts = np.bincount(ind)
    idxs = np.nonzero(counts)[0]
    prior_prob = np.zeros((q_ab.shape[0]))
    for i in range(q_ab.shape[0]):
        prior_prob[idxs] = counts[idxs]

    # We turn this into a color probability
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Save
    np.save(os.path.join(data_dir, "CelebA_%s_prior_prob.npy" % size), prior_prob)

    if do_plot:
        plt.hist(prior_prob, bins=100)
        plt.yscale("log")
        plt.show()


if __name__ == '__main__':
    data_dir = 'data/'
    do_plot = True

    X_ab = load_data()
    compute_color_prior(X_ab, size=64, do_plot=True)
