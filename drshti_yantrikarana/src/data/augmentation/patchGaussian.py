"""
Reference: 
Usage:

About: Script to create Patch Gaussian data augmentation method

Author: Satish Jasthi
"""

import tensorflow as tf
import numpy as np


def apply_patchGaussian(image: tf.image) -> tf.image:
    """
    Function to to create Patch Gaussian data augmentation method
    :param image: tf.image
    :return: tf.tensor
    """
    n_holes = 1
    length = image.shape[1] / 8

    image = np.array(image)
    h = image.shape[0]
    w = image.shape[1]

    mask = np.ones((h, w, 3), np.float32)

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = int(np.clip(y - length // 2, 0, h))
        y2 = int(np.clip(y + length // 2, 0, h))
        x1 = int(np.clip(x - length // 2, 0, w))
        x2 = int(np.clip(x + length // 2, 0, w))
        mask[y1: y2, x1: x2] = np.random.normal(loc=0, scale=np.random.uniform(low=0, high=0.1),
                                                size=mask[y1: y2, x1: x2].shape)
    image = image * mask
    return tf.convert_to_tensor(image)
