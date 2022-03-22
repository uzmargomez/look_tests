import cv2
import glob
import random
import numpy as np
import cvxpy as cp
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import sys

from lib.misc_functions import get_logger
from lib.Pipeline import detect_and_align

logger = get_logger()


class DataSet:
    def __init__(
        self,
        dir,
        ext,
        images_per_class,
        size,
        vertical,
        horizontal,
        epsilon,
        threshold,
        vis,
    ):

        self.test_images_known = {}
        self.test_images_unknown = {}
        self.test_images_group = []
        self.number_classes = 0
        self.classes = {}

        self.vis = vis
        self.size = size
        self.vertical = vertical
        self.horizontal = horizontal
        self.ver_pixels = int(size / vertical)
        self.hor_pixels = int(size / horizontal)
        self.images_per_class = images_per_class

        self.epsilon = epsilon
        self.threshold = threshold

        # Defining the different images for check accuracy

        aux = glob.glob(dir + "/Test/*")
        for directory in aux:
            images = glob.glob(directory + "/*." + ext)
            self.test_images_known[directory.split("/")[-1]] = images

        aux = glob.glob(dir + "/Unknown/*")
        for directory in aux:
            images = glob.glob(directory + "/*." + ext)
            self.test_images_unknown[directory.split("/")[-1]] = images

        aux = glob.glob(dir + "/Group")
        for directory in aux:
            images = glob.glob(directory + "/*." + ext)
            for i in images:
                self.test_images_group.append(i)

        A = [[] for i in range(vertical * horizontal)]
        all_subjects = glob.glob(dir + "/Train/*")

        for idx, directory in enumerate(all_subjects):
            logger.info(
                "Generating matrices: {:<6}%  |  Classes found: {:<3}".format(
                    round(100 * idx / len(all_subjects), 2),
                    self.number_classes,
                )
            )
            all_images = glob.glob(directory + "/*." + ext)[
                : self.images_per_class
            ]

            aligned_images = [
                detect_and_align(im, size, vis=self.vis) for im in all_images
            ]
            indices = [
                i for i, al in enumerate(aligned_images) if al is not None
            ]
            if not indices:
                continue

            self.classes[self.number_classes] = directory.split("/")[-1]
            self.number_classes += 1

            for aligned in aligned_images:
                matrices_index = 0
                for r in range(0, self.size, self.ver_pixels):
                    for c in range(0, self.size, self.hor_pixels):
                        cut = aligned[
                            r : r + self.ver_pixels, c : c + self.hor_pixels
                        ]
                        A[matrices_index].append(cut.flatten("F"))
                        matrices_index += 1
        self.matrices = [
            np.asmatrix(normalize(np.asmatrix(a).T, axis=0, norm="l2"))
            for a in A
        ]

    def classify(self, image, plot=False, vis=False):
        aligned = detect_and_align(image, self.size)
        if aligned is None:
            return None

        if vis:
            cv2.imshow("aligned", aligned)
            key = cv2.waitKey(0)

        Y = []  # Y = np.asmatrix(aligned.flatten("F")).T
        for r in range(0, self.size, self.ver_pixels):
            for c in range(0, self.size, self.hor_pixels):
                cut = aligned[r : r + self.ver_pixels, c : c + self.hor_pixels]
                Y.append(np.asmatrix(cut.flatten("F")).T)

        # X, e = optimization(self.matrix, Y, self.epsilon)
        X = [[] for i in range(self.vertical * self.horizontal)]
        e = [[] for i in range(self.vertical * self.horizontal)]
        votation = []

        for i in range(self.vertical * self.horizontal):
            X[i], e[i] = optimization(
                self.matrices[i], Y[i], self.epsilon, print_proc=False
            )

            delta_l = []
            for class_index in range(1, self.number_classes + 1):
                X_g = deltafunction(
                    class_index,
                    self.images_per_class,
                    self.number_classes,
                    X[i],
                )
                delta_l.append(X_g)

            e_r = []
            for class_index in range(0, self.number_classes):
                e_r.append(
                    np.linalg.norm(
                        Y[i] - e[i] - self.matrices[i] * delta_l[class_index],
                        2,
                    )
                )

            if plot:
                plt.plot(e_r, "o")
                plt.xlabel("Subject")
                plt.ylabel(r"Error $||y-A\delta_i||_2$")
                plt.grid()
                plt.savefig(f'residuals_{i}.png')

            if sci(X[i], delta_l) >= self.threshold:
                votation.append(np.argmin(e_r))
            else:
                votation.append(-1)

        most_common = int(max(votation, key=votation.count))

        if votation.count(most_common) < 2:
            return -1
        else:
            return most_common

    def set_threshold(self, threshold):
        self.threshold = threshold


def optimization(A, Y, epsilon, print_proc=False):
    """
    Solves the following optimization problem

       min      || w ||_1
        w


      subject to:

      Bw-Y <= epsilon

      where

      A = Matrix that contains info for all classes (subjects)
      X = Sparse coeficient vector, we look for it to be zero in
          most of its components, except for the ones
          representing one subject
      w = Concatenation of X(nx1) with an error vector e(mx1)
      B = Concatenation of matrix A(mxn) with the identity I(mxm)
      Y = Image to recognize

      ||.||_1   is called the l_1 norm
    """
    I = np.asmatrix(np.identity(A.shape[0]))

    B = np.concatenate((A, I), axis=1)

    x_rows = A.shape[1]

    # The variable, has to has the same rows as the matrix A
    w = cp.Variable((x_rows + A.shape[0], 1))

    # Objective function definition
    objective = cp.Minimize(cp.norm(w, 1))

    # Constraint definition
    constraints = [cp.norm(B @ w - Y, 2) <= epsilon]

    # Solve problem using cvxpy
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=print_proc)

    result = np.split(w.value, [x_rows, x_rows + A.shape[0]])

    X = np.asmatrix(result[0])
    e = np.asmatrix(result[1])

    return X, e


def deltafunction(class_index, images_per_class, number_classes, X):
    """
    Funtion that returns a vector of the same size as X
    whose only nonzero entries are the ones in X that are associated
    with "class_index"
    """

    m = images_per_class * number_classes

    d = np.asmatrix(np.zeros((m, 1)))

    X = np.asmatrix(X)

    for j in range(
        (class_index - 1) * images_per_class, class_index * images_per_class
    ):
        d[j, 0] = X[j, 0]

    return d


def sci(x, delta_l):
    """
    Sparsity Concentration Index (SCI)

    It measures how concentrated the coeficients are on a
    single class on the data set. It allows us to decide
    if the vector x is represented by the images of one subject
    on the dataset
    """

    norm_delta = []

    for i in delta_l:
        norm_delta.append(np.linalg.norm(i, 1))

    k = len(delta_l)

    return (k * max(norm_delta) / np.linalg.norm(x, 1) - 1) / (k - 1)
