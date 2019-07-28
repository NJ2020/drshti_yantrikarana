"""
Reference: 
Usage:

About: Script to create cutout data augmentation method

Author: Satish Jasthi
"""
import tensorflow as tf
import numpy as np


def apply_cutout(image:tf.image)->tf.image:
    """
    Function to create cutout data augmentation method
    :param image: tf.image
    :return: tf.tensor
    """
    n_holes = 1
    length = image.shape[1]/4

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
        mask[y1: y2, x1: x2] = 0.
    image = image * mask
    return tf.convert_to_tensor(image)

