"""
Usage:

About: Script to do

    - Pixel normalization
    - Image standardization

Author: Satish Jasthi
"""
import tensorflow as tf

from drshti_yantrikarana.config import data_mu, data_sigma

# TODO add test cases to both PixelNormalization and ImageStandardization
def PixelNormalization(image:tf.image)->tf.image:
    """
    Function to normalize image pixels by dividing
    every pixel by 255
    :param image: tf.image
    :return: tf.image
    """
    return image/255.0

def ImageStandardization(image:tf.image)->tf.image:
    """
    Function to standardize image by using doing
    (x - mu)/sigma
    where mu and sigma are channel wise mean and standard deviation
    :param image: tf.image
    :return: tf.image
    """
    tf.debugging.assert_type(data_mu, tf.float64)
    tf.debugging.assert_type(data_sigma, tf.float64)
    return (image-data_mu)/data_sigma