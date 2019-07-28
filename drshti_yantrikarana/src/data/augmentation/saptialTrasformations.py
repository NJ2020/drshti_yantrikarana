"""
Reference: 
Usage:

About: Script to create data augmentations that involve spatial transformations like
    - random rotate
    -  random flip

Author: Satish Jasthi
"""

import tensorflow as tf
import numpy as np
from PIL import Image

from drshti_yantrikarana.config import rotation_min, rotation_max


def apply_randomRotation(image: tf.image) -> tf.image:
    """
    Function to rotate image by random degree value
    :param image: tf.image
    :return: tf.tensor
    """
    image_arr = np.array(image)
    image = Image.fromarray(image_arr)
    rand_angle = np.random.randint(rotation_min, rotation_max)
    image_rot = Image.Image.rotate(image, angle=rand_angle)
    return tf.convert_to_tensor(np.array(image_rot))


def apply_flip(image: tf.image, mode='h') -> tf.image:
    """
    Function to flip an image either vertically or horizontally
    :param image:tf.image
    :param mode: str, 'v' for vertical flip and 'h' for horizontal flip
    :return: tf.tensor
    """
    if mode == 'v':
        return tf.image.flip_up_down(image)
    elif mode == 'h':
        return tf.image.flip_left_right(image)
