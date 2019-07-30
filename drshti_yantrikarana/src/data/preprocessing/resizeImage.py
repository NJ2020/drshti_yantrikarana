"""
Reference: 
Usage:

About: Module to resize images to defined resolution in config file


Author: Satish Jasthi
"""
import sys
from pathlib import Path

from PIL import Image
import tensorflow as tf

sys.path.append(Path(__file__).resolve().parent.parent.parent.parent.parent.as_posix())

from drshti_yantrikarana import resize_shape


def resize_image(img_tensor: tf.convert_to_tensor) -> tf.image:
    """
    Function to read an image, convert it into a square image and 
    resize it to standard resolution as defined in config
    :param image_path:tf.convert_to_tensor
    :return: tf.image
    """
    # create a central crop wrt larger side to create square image
    h, w = tf.shape(img_tensor)[:2]
    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(img_tensor,
                                                      offset_height=(h - w) // 2,
                                                      offset_width=0,
                                                      target_height=w,
                                                      target_width=w
                                                      )
    else:
        cropped_image = tf.image.crop_to_bounding_box(img_tensor,
                                                      offset_height=0,
                                                      offset_width=(w - h) // 2,
                                                      target_height=h,
                                                      target_width=h
                                                      )

    # resize image to predifined res in config
    resized_image = tf.image.resize(cropped_image, size=resize_shape)
    return resized_image
