import pickle
import numpy as np
import random
import os
from subprocess import call

random.seed(1)  # set a seed so that the results are consistent


def load_batch(test=False):
    path = 'cifar-10-batches-py/'
    if test:
        file = 'test_batch'
    else:
        file = 'data_batch_1'

    # Check if data is already downloaded
    if not os.path.exists(path + file):
        print("Downloading...")
        if not os.path.exists("cifar-10-python.tar.gz"):
            call(
                "curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                shell=True
            )
            print("Downloading done.\n")
        else:
            print("Dataset already downloaded. Did not download twice.\n")

        print("Extracting...")
        cifar_python_directory = os.path.abspath("cifar-10-batches-py")
        if not os.path.exists(cifar_python_directory):
            call(
                "tar -xvzf cifar-10-python.tar.gz",
                shell=True
            )
            print("Extracting successfully done to {}.".format(cifar_python_directory))
        else:
            print("Dataset already extracted. Did not extract twice.\n")

    f = open(path + file, 'rb')
    dict = pickle.load(f,encoding='latin1')
    images = dict['data']
    # images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imagearray = np.array(images)  # (10000, 3072)
    imagearray = np.rollaxis(np.reshape(imagearray, (10000, 3, 32, 32)), 1, 4) # (10000, 32, 32, 3)
    labelarray = np.array(labels)  # (10000,)

    return imagearray, labelarray


if __name__ == '__main__':
    load_batch()